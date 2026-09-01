#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::error::Result;
use datafusion::logical_expr::ScalarUDF;
use datafusion::prelude::{SessionConfig, SessionContext};
use liquid_cache::cache::squeeze_policies::{SqueezePolicy, TranscodeSqueezeEvict};
use liquid_cache::cache::{AlwaysHydrate, HydrationPolicy, default_max_memory_bytes};
use liquid_cache::cache_policies::{CachePolicy, LiquidPolicy};
use liquid_cache_datafusion::optimizers::{
    DEFAULT_MAX_LC_COLUMNS, LineageOptimizer, LocalModeOptimizer,
};
use liquid_cache_datafusion::{
    CacheEngagementPolicy, LiquidCacheParquet, VariantGetUdf, VariantPretty, VariantToJsonUdf,
    default_engagement_policy,
};

pub use liquid_cache as storage;
pub use liquid_cache_common as common;
pub use liquid_cache_datafusion::{LiquidCacheParquetRef, LiquidParquetSource};

/// Builder for in-process liquid cache session context
///
/// This allows you to use liquid cache within the same process,
/// instead of using the client-server architecture as in the default mode.
///
/// # Example
/// ```rust
/// use liquid_cache_datafusion_local::{
///     storage::cache_policies::LiquidPolicy,
///     LiquidCacheLocalBuilder,
/// };
/// use datafusion::prelude::{SessionConfig, SessionContext};
/// use tempfile::TempDir;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let temp_dir = TempDir::new().unwrap();
///
///     let (ctx, _) = LiquidCacheLocalBuilder::new()
///         .with_max_memory_bytes(1024 * 1024 * 1024) // 1GB
///         .with_cache_dir(temp_dir.path().to_path_buf())
///         .with_cache_policy(Box::new(LiquidPolicy::new()))
///         .build(SessionConfig::new())
///         .await?;
///
///     // Register the test parquet file
///     ctx.register_parquet("hits", "../../examples/nano_hits.parquet", Default::default())
///         .await?;
///
///     ctx.sql("SELECT COUNT(*) FROM hits").await?.show().await?;
///     Ok(())
/// }
/// ```
pub struct LiquidCacheLocalBuilder {
    /// Size of batches for caching
    batch_size: usize,
    /// Maximum memory size in bytes
    max_memory_bytes: usize,
    /// Maximum disk size in bytes
    max_disk_bytes: usize,
    /// Directory for disk cache
    cache_dir: PathBuf,
    /// Cache policy
    cache_policy: Box<dyn CachePolicy>,
    /// Squeeze policy
    squeeze_policy: Box<dyn SqueezePolicy>,
    /// Hydration policy
    hydration_policy: Box<dyn HydrationPolicy>,
    /// Whether to register the lineage (date/variant EXTRACT) logical optimizer.
    /// Enabled by default; disable to skip its per-query planning cost when the
    /// workload never benefits from date-component/variant squeezing.
    enable_lineage_optimizer: bool,
    /// Maximum number of projected output columns for which a scan is wrapped
    /// with liquid cache. Wider projections are left as plain parquet scans.
    max_projected_columns: usize,
    /// Per-file cache engagement policy (e.g. selectivity threshold).
    engagement_policy: Arc<dyn CacheEngagementPolicy>,
    span: fastrace::Span,
}

impl Default for LiquidCacheLocalBuilder {
    fn default() -> Self {
        let max_memory_bytes = default_max_memory_bytes();
        let max_disk_bytes = max_memory_bytes.saturating_mul(10);
        Self {
            batch_size: 8192,
            max_memory_bytes,
            max_disk_bytes,
            cache_dir: std::env::temp_dir(),
            cache_policy: Box::new(LiquidPolicy::new()),
            squeeze_policy: Box::new(TranscodeSqueezeEvict),
            hydration_policy: Box::new(AlwaysHydrate::new()),
            enable_lineage_optimizer: true,
            max_projected_columns: DEFAULT_MAX_LC_COLUMNS,
            engagement_policy: default_engagement_policy(),
            span: fastrace::Span::enter_with_local_parent("liquid_cache_datafusion_local_builder"),
        }
    }
}

impl LiquidCacheLocalBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set maximum memory size in bytes.
    /// Default is half of available system memory.
    pub fn with_max_memory_bytes(mut self, max_memory_bytes: usize) -> Self {
        self.max_memory_bytes = max_memory_bytes;
        self
    }

    /// Set maximum disk size in bytes.
    /// Default is 10x the default memory size.
    pub fn with_max_disk_bytes(mut self, max_disk_bytes: usize) -> Self {
        self.max_disk_bytes = max_disk_bytes;
        self
    }

    /// Set cache directory
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Set squeeze policy
    pub fn with_squeeze_policy(mut self, squeeze_policy: Box<dyn SqueezePolicy>) -> Self {
        self.squeeze_policy = squeeze_policy;
        self
    }

    /// Set cache strategy
    pub fn with_cache_policy(mut self, cache_policy: Box<dyn CachePolicy>) -> Self {
        self.cache_policy = cache_policy;
        self
    }

    /// Set hydration policy
    pub fn with_hydration_policy(mut self, hydration_policy: Box<dyn HydrationPolicy>) -> Self {
        self.hydration_policy = hydration_policy;
        self
    }

    /// Enable or disable the lineage (date/variant `EXTRACT`) logical optimizer.
    ///
    /// Enabled by default to match upstream behavior. Disabling skips registering
    /// the rule, avoiding its per-query planning cost. This is safe for workloads
    /// that do not rely on date-component or variant-field squeezing (e.g. plain
    /// numeric/date/boolean projections); results are unaffected either way.
    pub fn with_lineage_optimizer(mut self, enable: bool) -> Self {
        self.enable_lineage_optimizer = enable;
        self
    }

    /// Set the maximum number of projected output columns for which a scan is
    /// wrapped with liquid cache. Scans projecting more columns are left as plain
    /// parquet scans. Defaults to [`DEFAULT_MAX_LC_COLUMNS`].
    pub fn with_max_projected_columns(mut self, max_projected_columns: usize) -> Self {
        self.max_projected_columns = max_projected_columns;
        self
    }

    /// Set the per-file cache engagement policy (e.g. a selectivity threshold that
    /// delegates low-selectivity predicate scans to plain parquet). Defaults to
    /// [`default_engagement_policy`].
    pub fn with_engagement_policy(mut self, policy: Arc<dyn CacheEngagementPolicy>) -> Self {
        self.engagement_policy = policy;
        self
    }

    /// Set fastrace span
    pub fn with_span(mut self, span: fastrace::Span) -> Self {
        self.span = span;
        self
    }

    /// Build a SessionContext with liquid cache configured
    /// Returns the SessionContext and the liquid cache reference
    pub async fn build(
        self,
        mut config: SessionConfig,
    ) -> Result<(SessionContext, LiquidCacheParquetRef)> {
        config.options_mut().execution.parquet.pushdown_filters = true;
        config
            .options_mut()
            .execution
            .parquet
            .schema_force_view_types = false;
        config.options_mut().execution.parquet.skip_arrow_metadata = false;
        config.options_mut().execution.parquet.skip_metadata = false;
        config.options_mut().execution.batch_size = self.batch_size;

        // Disk tier: O_DIRECT + O_DSYNC on Linux (production default). Other
        // platforms fall back to buffered I/O — macOS/Windows have no O_DIRECT
        // — which changes durability characteristics of the disk tier only,
        // never correctness; the in-memory cache is identical everywhere.
        let mount_options = t4::MountOptions {
            direct_io: cfg!(target_os = "linux"),
            ..Default::default()
        };
        let store = t4::mount_with_options(self.cache_dir.join("liquid_cache.t4"), mount_options)
            .await
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
        #[cfg(not(test))]
        let cache = LiquidCacheParquet::new(
            self.batch_size,
            self.max_memory_bytes,
            self.max_disk_bytes,
            store,
            self.cache_policy,
            self.squeeze_policy,
            self.hydration_policy,
        )
        .await;

        #[cfg(test)]
        let cache = LiquidCacheParquet::new_with_squeeze_victim_concurrency(
            self.batch_size,
            self.max_memory_bytes,
            self.max_disk_bytes,
            store,
            self.cache_policy,
            self.squeeze_policy,
            self.hydration_policy,
            false,
        )
        .await;
        let cache_ref = Arc::new(cache);

        let optimizer = LocalModeOptimizer::new(cache_ref.clone())
            .with_max_projected_columns(self.max_projected_columns)
            .with_engagement_policy(self.engagement_policy.clone());

        let mut state_builder = datafusion::execution::SessionStateBuilder::new()
            .with_config(config)
            .with_default_features();
        if self.enable_lineage_optimizer {
            state_builder = state_builder.with_optimizer_rule(Arc::new(LineageOptimizer::new()));
        }
        let state = state_builder
            .with_physical_optimizer_rule(Arc::new(optimizer))
            .build();

        let ctx = SessionContext::new_with_state(state);
        ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantPretty::default()));
        ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
        Ok((ctx, cache_ref))
    }
}

#[cfg(test)]
mod local_tests {
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTableUrl},
    };

    use super::*;

    #[tokio::test]
    async fn register_with_listing_table() -> Result<()> {
        let file_format = ParquetFormat::default().with_enable_pruning(true);
        let listing_options =
            ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");
        let (ctx, _) = LiquidCacheLocalBuilder::new()
            .build(SessionConfig::new())
            .await?;
        let table_path = ListingTableUrl::parse("../../examples/nano_hits.parquet")?;
        ctx.register_listing_table("hits", &table_path, listing_options.clone(), None, None)
            .await?;

        ctx.sql("SELECT * FROM hits where \"URL\" like '%google%'")
            .await?
            .show()
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_provide_schema() -> Result<()> {
        let (ctx, _) = LiquidCacheLocalBuilder::new()
            .build(SessionConfig::new())
            .await?;

        let file_format = ParquetFormat::default().with_enable_pruning(true);
        let listing_options =
            ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");

        let table_path = ListingTableUrl::parse("../../examples/nano_hits.parquet")?;
        let schema = Schema::new(vec![Field::new("WatchID", DataType::Int64, false)]);

        ctx.register_listing_table(
            "hits",
            &table_path,
            listing_options.clone(),
            Some(Arc::new(schema)),
            None,
        )
        .await?;

        ctx.sql("SELECT \"WatchID\" FROM hits limit 1")
            .await?
            .show()
            .await?;
        Ok(())
    }
}
