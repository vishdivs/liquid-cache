//! Optimizers for the Parquet module

mod lineage_opt;

use std::sync::Arc;

use arrow_schema::{Field, Schema, SchemaRef};
use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSource,
        table_schema::TableSchema,
    },
    physical_expr_adapter::PhysicalExprAdapterFactory,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};
pub use lineage_opt::LineageOptimizer;

use crate::{
    CacheEngagementPolicy, LiquidCacheParquetRef, LiquidParquetSource, default_engagement_policy,
    optimizers::lineage_opt::{ColumnAnnotation, metadata_from_factory, serialize_date_part},
};

/// Default maximum number of projected output columns for which LC wrapping is
/// applied. Above this, per-column cache overhead tends to exceed decode savings,
/// so the scan is left as a plain `ParquetSource`.
pub const DEFAULT_MAX_LC_COLUMNS: usize = 4;

pub(crate) const DATE_MAPPING_METADATA_KEY: &str = "liquid.cache.date_mapping";
pub(crate) const STRING_FINGERPRINT_METADATA_KEY: &str = "liquid.cache.string_fingerprint";

/// Physical optimizer rule for local mode liquid cache
///
/// This optimizer rewrites DataSourceExec nodes that read Parquet files
/// to use LiquidParquetSource instead of the default ParquetSource
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
    max_projected_columns: usize,
    engagement_policy: Arc<dyn CacheEngagementPolicy>,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            max_projected_columns: DEFAULT_MAX_LC_COLUMNS,
            engagement_policy: default_engagement_policy(),
        }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self::new(cache)
    }

    /// Set the maximum number of projected output columns for which the scan is
    /// wrapped with liquid cache. Scans projecting more columns are left as plain
    /// `ParquetSource`. Defaults to [`DEFAULT_MAX_LC_COLUMNS`].
    pub fn with_max_projected_columns(mut self, max_projected_columns: usize) -> Self {
        self.max_projected_columns = max_projected_columns;
        self
    }

    /// Set the cache engagement policy applied to wrapped scans at file-open time.
    /// Defaults to [`default_engagement_policy`].
    pub fn with_engagement_policy(mut self, policy: Arc<dyn CacheEngagementPolicy>) -> Self {
        self.engagement_policy = policy;
        self
    }
}

impl PhysicalOptimizerRule for LocalModeOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, datafusion::error::DataFusionError> {
        Ok(rewrite_data_source_plan_with_config(
            plan,
            &self.cache,
            self.max_projected_columns,
            &self.engagement_policy,
        ))
    }

    fn name(&self) -> &str {
        "LocalModeLiquidCacheOptimizer"
    }

    fn schema_check(&self) -> bool {
        // We deliberately enrich scan schemas with metadata describing variant/date
        // extractions, so allow the optimizer to adjust schema metadata.
        false
    }
}

/// Rewrite the data source plan to use liquid cache, using default configuration
/// ([`DEFAULT_MAX_LC_COLUMNS`] and [`default_engagement_policy`]).
pub fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
) -> Arc<dyn ExecutionPlan> {
    let engagement_policy = default_engagement_policy();
    rewrite_data_source_plan_with_config(plan, cache, DEFAULT_MAX_LC_COLUMNS, &engagement_policy)
}

/// Rewrite the data source plan to use liquid cache with explicit configuration.
pub fn rewrite_data_source_plan_with_config(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    max_projected_columns: usize,
    engagement_policy: &Arc<dyn CacheEngagementPolicy>,
) -> Arc<dyn ExecutionPlan> {
    let rewritten = plan
        .transform_up(|node| {
            try_optimize_parquet_source(node, cache, max_projected_columns, engagement_policy)
        })
        .unwrap();
    rewritten.data
}

/// Returns true if a data type is uncacheable by LC (string/binary).
fn is_uncacheable_type(dt: &arrow_schema::DataType) -> bool {
    use arrow_schema::DataType;
    matches!(
        dt,
        DataType::Utf8
            | DataType::Utf8View
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::BinaryView
            | DataType::LargeBinary
    ) || matches!(dt, DataType::Dictionary(_, v) if is_uncacheable_type(v))
}

fn try_optimize_parquet_source(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    max_projected_columns: usize,
    engagement_policy: &Arc<dyn CacheEngagementPolicy>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>, datafusion::error::DataFusionError> {
    if let Some(data_source_exec) = plan.downcast_ref::<DataSourceExec>()
        && let Some((file_scan_config, parquet_source)) =
            data_source_exec.downcast_to_file_source::<ParquetSource>()
    {
        // Skip LC wrapping if:
        //   - Output has zero columns (COUNT(*) — just needs row count from metadata)
        //   - ANY output column is string/binary (LC can't cache, fallback negates hits)
        //   - Predicate references a string column
        let output_schema = plan.schema();
        if output_schema.fields().is_empty() {
            log::debug!("[LC-Optimizer] SKIP: empty projection (COUNT(*))");
            return Ok(Transformed::no(plan));
        }

        // Skip LC when too many output columns — per-column cache overhead
        // exceeds decode savings for wide projections.
        if output_schema.fields().len() > max_projected_columns {
            log::debug!(
                "[LC-Optimizer] SKIP: too many columns ({} > {})",
                output_schema.fields().len(),
                max_projected_columns
            );
            return Ok(Transformed::no(plan));
        }

        let has_string_output = output_schema
            .fields()
            .iter()
            .any(|f| is_uncacheable_type(f.data_type()));

        let predicate_has_string = parquet_source.filter().map_or(false, |pred| {
            use datafusion::physical_expr::utils::collect_columns;
            let file_schema = file_scan_config.file_schema();
            let cols = collect_columns(&pred);
            cols.iter().any(|col| {
                file_schema
                    .fields()
                    .get(col.index())
                    .map_or(false, |f| is_uncacheable_type(f.data_type()))
            })
        });

        if has_string_output || predicate_has_string {
            log::debug!(
                "[LC-Optimizer] SKIP: string_in_output={}, string_in_predicate={}, output_cols={}",
                has_string_output, predicate_has_string, output_schema.fields().len()
            );
            return Ok(Transformed::no(plan));
        }

        let num_fields = output_schema.fields().len();
        let has_predicate = parquet_source.filter().is_some();
        log::debug!(
            "[LC-Optimizer] WRAP: all {} output columns cacheable, predicate={}",
            num_fields, has_predicate
        );

        let mut new_config = file_scan_config.clone();

        let mut new_source =
            LiquidParquetSource::from_parquet_source(parquet_source.clone(), cache.clone())
                .with_engagement_policy(Arc::clone(engagement_policy));
        if let Some(expr_adapter_factory) = file_scan_config.expr_adapter_factory.as_ref() {
            let new_schema =
                enrich_source_schema(file_scan_config.file_schema(), expr_adapter_factory);
            let table_partition_cols = new_source.table_schema().table_partition_cols();
            let new_table_schema =
                TableSchema::new(Arc::new(new_schema), table_partition_cols.clone());
            new_source = new_source.with_table_schema(new_table_schema);
        }

        new_config.file_source = Arc::new(new_source);
        let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
        let new_plan = Arc::new(DataSourceExec::new(new_file_source));

        return Ok(Transformed::new(
            new_plan,
            true,
            TreeNodeRecursion::Continue,
        ));
    }
    Ok(Transformed::no(plan))
}

fn enrich_source_schema(
    file_schema: &SchemaRef,
    expr_adapter_factory: &Arc<dyn PhysicalExprAdapterFactory>,
) -> Schema {
    let mut new_fields = vec![];
    for field in file_schema.fields() {
        if let Some(annotation) = metadata_from_factory(expr_adapter_factory, field.name()) {
            new_fields.push(process_field_annotation(field, annotation));
        } else {
            new_fields.push(field.clone());
        }
    }
    Schema::new(new_fields)
}

fn process_field_annotation(field: &Arc<Field>, annotation: ColumnAnnotation) -> Arc<Field> {
    let mut field_metadata = field.metadata().clone();
    match annotation {
        ColumnAnnotation::DatePart(unit) => {
            field_metadata.insert(
                DATE_MAPPING_METADATA_KEY.to_string(),
                serialize_date_part(&unit),
            );
        }
        ColumnAnnotation::VariantPaths(_) => {}
        ColumnAnnotation::SubstringSearch => {
            field_metadata.insert(
                STRING_FINGERPRINT_METADATA_KEY.to_string(),
                "substring".into(),
            );
        }
    }
    Arc::new(Field::clone(field.as_ref()).with_metadata(field_metadata))
}

#[cfg(test)]
mod tests {
    use datafusion::{datasource::physical_plan::FileScanConfig, prelude::SessionContext};
    use liquid_cache::{
        cache::{AlwaysHydrate, squeeze_policies::TranscodeSqueezeEvict},
        cache_policies::LiquidPolicy,
    };

    use crate::LiquidCacheParquet;

    use super::*;

    /// Extract the file schema of the (first) parquet scan in a plan.
    fn scan_file_schema(plan: &Arc<dyn ExecutionPlan>) -> SchemaRef {
        let mut schema = None;
        plan.apply(|node| {
            if let Some(exec) = node.downcast_ref::<DataSourceExec>()
                && let Some(cfg) = exec.data_source().downcast_ref::<FileScanConfig>()
            {
                schema = Some(cfg.file_schema().clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        schema.expect("plan should contain a parquet file scan")
    }

    async fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) {
        // Capture the scan's file schema before the rewrite so we can assert the
        // optimizer wraps the scan without corrupting its file schema.
        let original_file_schema = scan_file_schema(&plan);
        let liquid_cache = build_test_cache().await;
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache);

        let mut wrapped = false;
        rewritten
            .apply(|node| {
                if let Some(exec) = node.downcast_ref::<DataSourceExec>() {
                    let source = exec.data_source();
                    let cfg = source.downcast_ref::<FileScanConfig>().unwrap();
                    // The scan must be wrapped with liquid cache ...
                    cfg.file_source()
                        .downcast_ref::<LiquidParquetSource>()
                        .unwrap();
                    wrapped = true;
                    // ... and the file schema must be preserved unchanged.
                    assert_eq!(cfg.file_schema().as_ref(), original_file_schema.as_ref());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        assert!(
            wrapped,
            "expected scan to be wrapped in LiquidParquetSource"
        );
    }

    async fn build_test_cache() -> LiquidCacheParquetRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        // direct_io (O_DIRECT) is Linux-only; fall back to buffered I/O elsewhere
        // so the test runs on macOS/Windows too.
        let mount_options = t4::MountOptions {
            direct_io: cfg!(target_os = "linux"),
            ..Default::default()
        };
        let store = t4::mount_with_options(tmp_dir.path().join("liquid_cache.t4"), mount_options)
            .await
            .unwrap();
        Arc::new(
            LiquidCacheParquet::new(
                8192,
                1000000,
                usize::MAX,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(TranscodeSqueezeEvict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        )
    }

    fn has_liquid_source(plan: &Arc<dyn ExecutionPlan>) -> bool {
        let mut found = false;
        plan.apply(|node| {
            if let Some(exec) = node.downcast_ref::<DataSourceExec>()
                && let Some(cfg) = exec.data_source().downcast_ref::<FileScanConfig>()
                && cfg
                    .file_source()
                    .downcast_ref::<LiquidParquetSource>()
                    .is_some()
            {
                found = true;
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    /// The max-projected-columns knob gates whether a scan is wrapped. With a cap
    /// of 0, any non-empty projection is left as a plain parquet scan; with the
    /// default cap the same plan is wrapped.
    #[tokio::test]
    async fn max_projected_columns_gates_wrapping() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        let df = ctx
            .sql("SELECT \"WatchID\" FROM nano_hits LIMIT 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let cache = build_test_cache().await;
        let engagement = default_engagement_policy();

        // Cap of 0: the single-column projection exceeds the cap, so no wrapping.
        let capped = rewrite_data_source_plan_with_config(plan.clone(), &cache, 0, &engagement);
        assert!(
            !has_liquid_source(&capped),
            "max_projected_columns=0 must leave the scan as plain parquet"
        );

        // Default cap: the same plan is wrapped with liquid cache.
        let wrapped = rewrite_data_source_plan(plan, &cache);
        assert!(
            has_liquid_source(&wrapped),
            "default cap must wrap a narrow numeric projection"
        );
    }

    #[tokio::test]
    async fn test_plan_rewrite() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        // Narrow numeric projection so the scan qualifies for LC wrapping; a wide
        // `SELECT *` with string columns is intentionally skipped by the optimizer.
        let df = ctx
            .sql("SELECT \"WatchID\" FROM nano_hits WHERE \"WatchID\" > 0 limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        rewrite_plan_inner(plan.clone()).await;
    }
}
