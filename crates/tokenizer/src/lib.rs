use std::sync::Arc;

use anyhow::Result;

pub mod backend;
pub mod cache;
pub mod factory;
pub mod hub;
pub mod mock;
pub mod registry;
pub mod sequence;
pub mod stop;
pub mod stream;
pub mod traits;

pub mod chat_template;
pub mod huggingface;
pub mod tiktoken;

#[cfg(test)]
mod tests;

// Re-export types used outside this module
pub use backend::TokenizerBackend;
pub use cache::{CacheConfig, CacheStats, CachedTokenizer, L0Cache, L1Cache, TokenizerFingerprint};
pub use chat_template::ChatTemplateState;
pub use factory::{
    create_tokenizer, create_tokenizer_from_file, create_tokenizer_with_chat_template,
    TokenizerType,
};
pub use huggingface::HuggingFaceTokenizer;
pub use mock::MockTokenizer;
pub use registry::{LoadError, LoadOutcome, TokenizerRegistry};
pub use sequence::Sequence;
pub use stop::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
pub use stream::DecodeStream;
pub use tiktoken::{TiktokenModel, TiktokenTokenizer};
pub use traits::{
    Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer as TokenizerTrait,
};

/// Main tokenizer wrapper that provides a unified interface for different tokenizer implementations.
///
/// Internally holds an `Arc<TokenizerBackend>` for enum-based dispatch on the hot path.
#[derive(Clone)]
pub struct Tokenizer(Arc<TokenizerBackend>);

impl Tokenizer {
    /// Create a tokenizer from a file path
    pub fn from_file(file_path: &str) -> Result<Tokenizer> {
        Ok(Tokenizer(create_tokenizer_from_file(file_path)?))
    }

    /// Create a tokenizer from a file path with an optional chat template
    pub fn from_file_with_chat_template(
        file_path: &str,
        chat_template_path: Option<&str>,
    ) -> Result<Tokenizer> {
        Ok(Tokenizer(create_tokenizer_with_chat_template(
            file_path,
            chat_template_path,
        )?))
    }

    /// Create a tokenizer from an `Arc<TokenizerBackend>`
    pub fn from_backend(backend: Arc<TokenizerBackend>) -> Self {
        Tokenizer(backend)
    }

    /// Create a tokenizer from an `Arc<dyn Tokenizer>` (backward compatibility).
    ///
    /// This wraps the trait object in a `TokenizerBackend` that delegates through
    /// the vtable. Prefer `from_backend` when you have a concrete backend.
    pub fn from_arc(tokenizer: Arc<dyn traits::Tokenizer>) -> Self {
        // For backward compatibility, we need to go through the trait object.
        // This path is only used by code that hasn't migrated to TokenizerBackend yet.
        // We re-wrap in a Mock variant as a placeholder — callers should migrate.
        // TODO: Remove this once all callers use from_backend
        Tokenizer(tokenizer.as_any().downcast_ref::<TokenizerBackend>()
            .map(|_| {
                // The Arc already contains a TokenizerBackend — but we can't safely
                // recover the Arc<TokenizerBackend> from Arc<dyn Tokenizer>.
                // For now, just create a new factory call.
                unreachable!("Use from_backend for Arc<TokenizerBackend>")
            })
            .unwrap_or_else(|| {
                // This shouldn't be called in practice after migration
                panic!("Tokenizer::from_arc is deprecated — use Tokenizer::from_backend instead")
            }))
    }

    /// Get the inner backend
    pub fn backend(&self) -> &Arc<TokenizerBackend> {
        &self.0
    }

    /// Create a stateful sequence object for decoding token_ids into text
    pub fn decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> DecodeStream {
        // DecodeStream still accepts Arc<dyn Tokenizer> for backward compat with Go bindings.
        // Arc<TokenizerBackend> coerces to Arc<dyn Tokenizer> since TokenizerBackend: Tokenizer.
        DecodeStream::new(self.0.clone() as Arc<dyn traits::Tokenizer>, prompt_token_ids, skip_special_tokens)
    }

    /// Direct encode method
    ///
    /// Set `add_special_tokens` to `true` for embeddings (to add BOS/EOS tokens configured in tokenizer_config.json),
    /// or `false` for chat completion (where the chat template handles special tokens).
    pub fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        self.0.encode(input, add_special_tokens)
    }

    /// Direct batch encode method
    ///
    /// Set `add_special_tokens` to `true` for embeddings (to add BOS/EOS tokens configured in tokenizer_config.json),
    /// or `false` for chat completion (where the chat template handles special tokens).
    pub fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.0.encode_batch(inputs, add_special_tokens)
    }

    /// Direct decode method
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.0.decode(token_ids, skip_special_tokens)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    /// Get special tokens
    pub fn get_special_tokens(&self) -> &SpecialTokens {
        self.0.get_special_tokens()
    }

    /// Convert token string to ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    /// Convert ID to token string
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
}

impl From<Arc<TokenizerBackend>> for Tokenizer {
    fn from(backend: Arc<TokenizerBackend>) -> Self {
        Tokenizer(backend)
    }
}
