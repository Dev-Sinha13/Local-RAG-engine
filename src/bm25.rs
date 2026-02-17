/// BM25 (Okapi BM25) search index for keyword-based document retrieval.
///
/// Implements the standard BM25 ranking function:
///   score(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
///
/// Built entirely in Rust for performance when scoring thousands of chunks.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::tokenizer;

/// A BM25 search index built from a collection of text documents.
///
/// Construct from Python with:
///     index = BM25Index(["chunk 1 text", "chunk 2 text", ...])
///     results = index.search("my query", top_k=5)
#[pyclass]
pub struct BM25Index {
    /// Term → number of documents containing it
    df: HashMap<String, usize>,
    /// Per-document term frequencies
    tf: Vec<HashMap<String, usize>>,
    /// Token count per document
    doc_lengths: Vec<usize>,
    /// Average document length
    avg_dl: f64,
    /// Total number of documents
    n_docs: usize,
    /// BM25 tuning parameters
    k1: f64,
    b: f64,
}

#[pymethods]
impl BM25Index {
    /// Build a BM25 index from a list of document strings.
    ///
    /// Args:
    ///     documents: List of text strings to index.
    ///     k1: Term frequency saturation parameter (default 1.2).
    ///     b: Length normalization parameter (default 0.75).
    #[new]
    #[pyo3(signature = (documents, k1=1.2, b=0.75))]
    fn new(documents: Vec<String>, k1: f64, b: f64) -> Self {
        let n_docs = documents.len();
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut tf: Vec<HashMap<String, usize>> = Vec::with_capacity(n_docs);
        let mut doc_lengths: Vec<usize> = Vec::with_capacity(n_docs);

        for doc in &documents {
            let tokens = tokenizer::tokenize(doc);
            doc_lengths.push(tokens.len());

            let mut term_freq: HashMap<String, usize> = HashMap::new();
            let mut seen: HashSet<String> = HashSet::new();

            for token in &tokens {
                *term_freq.entry(token.clone()).or_insert(0) += 1;
                if seen.insert(token.clone()) {
                    *df.entry(token.clone()).or_insert(0) += 1;
                }
            }

            tf.push(term_freq);
        }

        let avg_dl = if n_docs > 0 {
            doc_lengths.iter().sum::<usize>() as f64 / n_docs as f64
        } else {
            0.0
        };

        BM25Index {
            df,
            tf,
            doc_lengths,
            avg_dl,
            n_docs,
            k1,
            b,
        }
    }

    /// Score all documents against the query and return top-k results.
    ///
    /// Returns a list of (document_index, score) tuples, sorted by
    /// score descending. Only documents with score > 0 are returned.
    #[pyo3(signature = (query, top_k=10))]
    fn search(&self, query: &str, top_k: usize) -> Vec<(usize, f64)> {
        let query_tokens = tokenizer::tokenize(query);
        let mut scores: Vec<(usize, f64)> = Vec::new();

        for (doc_idx, doc_tf) in self.tf.iter().enumerate() {
            let doc_len = self.doc_lengths[doc_idx] as f64;
            let mut score = 0.0;

            for token in &query_tokens {
                let tf = *doc_tf.get(token).unwrap_or(&0) as f64;
                let df = *self.df.get(token).unwrap_or(&0) as f64;

                if tf == 0.0 {
                    continue;
                }

                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((self.n_docs as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();

                // TF with length normalization
                let tf_norm = (tf * (self.k1 + 1.0))
                    / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_dl));

                score += idf * tf_norm;
            }

            if score > 0.0 {
                scores.push((doc_idx, score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Return the number of indexed documents.
    fn __len__(&self) -> usize {
        self.n_docs
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        format!(
            "BM25Index(n_docs={}, vocab_size={}, avg_dl={:.1}, k1={}, b={})",
            self.n_docs,
            self.df.len(),
            self.avg_dl,
            self.k1,
            self.b
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_index() {
        let docs = vec![
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
            "the cat chased the dog".to_string(),
        ];
        let index = BM25Index::new(docs, 1.2, 0.75);
        assert_eq!(index.n_docs, 3);
        assert_eq!(index.doc_lengths, vec![6, 6, 5]);
    }

    #[test]
    fn test_search_relevance() {
        let docs = vec![
            "machine learning and deep learning".to_string(),
            "cooking recipes and food preparation".to_string(),
            "neural networks for machine learning".to_string(),
        ];
        let index = BM25Index::new(docs, 1.2, 0.75);
        let results = index.search("machine learning", 3);

        // Docs 0 and 2 should rank higher than doc 1
        assert!(!results.is_empty());
        assert!(results[0].0 == 0 || results[0].0 == 2);
        // Doc 1 (cooking) should not appear or rank last
        assert!(results.iter().all(|&(idx, _)| idx != 1));
    }

    #[test]
    fn test_search_no_match() {
        let docs = vec![
            "the cat sat on the mat".to_string(),
            "the dog sat on the log".to_string(),
        ];
        let index = BM25Index::new(docs, 1.2, 0.75);
        let results = index.search("quantum physics", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_index() {
        let index = BM25Index::new(vec![], 1.2, 0.75);
        let results = index.search("anything", 5);
        assert!(results.is_empty());
        assert_eq!(index.n_docs, 0);
    }

    #[test]
    fn test_top_k_limit() {
        let docs: Vec<String> = (0..20)
            .map(|i| format!("document number {} about rust programming", i))
            .collect();
        let index = BM25Index::new(docs, 1.2, 0.75);
        let results = index.search("rust programming", 5);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_more_matches_score_higher() {
        let docs = vec![
            "rust programming language systems".to_string(),        // has: rust, programming, systems
            "python scripting language interpreted".to_string(),    // has: none of query terms
            "rust is great for systems programming".to_string(),   // has: rust, programming, systems
        ];
        let index = BM25Index::new(docs, 1.2, 0.75);
        let results = index.search("rust systems programming", 3);

        // Docs 0 and 2 have all query terms, doc 1 has none
        assert!(results.len() >= 2);
        let top_indices: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(top_indices.contains(&0));
        assert!(top_indices.contains(&2));
    }
}
