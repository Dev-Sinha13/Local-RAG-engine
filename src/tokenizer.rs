/// Word-level tokenizer for text processing and BM25 scoring.
///
/// Splits on non-alphanumeric characters (preserving apostrophes for
/// contractions like "don't"), lowercases everything, and filters empties.

/// Tokenize text into lowercase word tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Count the number of word tokens in text.
pub fn token_count(text: &str) -> usize {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_apostrophes_preserved() {
        let tokens = tokenize("don't won't can't it's");
        assert_eq!(tokens, vec!["don't", "won't", "can't", "it's"]);
    }

    #[test]
    fn test_empty_input() {
        assert!(tokenize("").is_empty());
        assert_eq!(token_count(""), 0);
    }

    #[test]
    fn test_special_characters() {
        let tokens = tokenize("hello---world...test!!!end");
        assert_eq!(tokens, vec!["hello", "world", "test", "end"]);
    }

    #[test]
    fn test_mixed_case() {
        let tokens = tokenize("GPT-4 BERT transformer");
        assert_eq!(tokens, vec!["gpt", "4", "bert", "transformer"]);
    }

    #[test]
    fn test_token_count() {
        assert_eq!(token_count("Hello World"), 2);
        assert_eq!(token_count("one"), 1);
        assert_eq!(token_count("a b c d e"), 5);
    }

    #[test]
    fn test_numbers() {
        let tokens = tokenize("chapter 3.14 section 2");
        assert_eq!(tokens, vec!["chapter", "3", "14", "section", "2"]);
    }
}
