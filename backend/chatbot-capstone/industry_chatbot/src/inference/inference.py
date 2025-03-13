#!/usr/bin/env python3
import json
import numpy as np
import faiss
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class PolicyQASystem:
    """
    A policy question-answering system that uses semantic search over a knowledge base
    and a generative model as a fallback. When retrieval confidence is low, the best
    available context is fed into the generative prompt so that the answer remains anchored
    in the knowledge base.
    """
    
    def __init__(
        self, 
        data_path: str, 
        retrieval_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gen_model: str = "EleutherAI/gpt-neo-125M",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the PolicyQASystem with data, models, and optional logging.
        
        Args:
            data_path (str): Path to the JSON file containing QA pairs.
            retrieval_model (str): Sentence transformer model for semantic search.
            gen_model (str): Language model for answer generation.
            logger (Optional[logging.Logger]): Custom logger for tracking operations.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Load QA data
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        
        # Prepare QA data: extract questions and answers.
        self.questions = [item["question"] for item in self.data]
        self.answers = {item["question"]: item["answer"] for item in self.data}
        
        # Initialize models.
        try:
            self.retrieval_model = SentenceTransformer(retrieval_model)
            self.tokenizer = AutoTokenizer.from_pretrained(gen_model)
            self.gen_model = AutoModelForCausalLM.from_pretrained(gen_model)
            
            # Ensure tokenizer has a pad token.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.gen_model.config.pad_token_id = self.tokenizer.eos_token_id
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
            raise
        
        # Build a semantic search index using FAISS.
        self._build_index()
        
        # Define some templates for direct retrieval formatting.
        self.templates = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?"
            ],
            'customer_satisfaction': [
                "Our customer satisfaction approach ensures {} while prioritizing {}",
                "We systematically implement customer satisfaction through {} and {}",
                "Our commitment to excellence is demonstrated by {} and {}"
            ],
            'it_policy': [
                "Regarding IT infrastructure, our policy mandates that {}",
                "Our technology governance ensures that {}",
                "To maintain operational excellence, we implement policies that {}"
            ],
            'default': [
                "According to our organizational policy, {}",
                "Our structured guidelines specify that {}",
                "We adhere to the principle that {}"
            ]
        }
        
        # Simple greetings for quick response.
        self.greeting_words = {"hi", "hello", "hey", "greetings"}
    
    def _build_index(self) -> None:
        """
        Build a semantic search index using FAISS for efficient retrieval.
        """
        try:
            self.question_embeddings = self.retrieval_model.encode(
                self.questions, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            # Normalize embeddings for cosine similarity (using inner product on L2-normalized vectors).
            faiss.normalize_L2(self.question_embeddings)
            dim = self.question_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.question_embeddings)
        except Exception as e:
            self.logger.error(f"Index building error: {e}")
            raise
    
    def _select_template(self, query: str, policy_type: str = 'default') -> str:
        """
        Select a formatting template based on query context.
        """
        templates = self.templates.get(policy_type, self.templates['default'])
        return np.random.choice(templates)
    
    def _format_policy_answer(self, answer: str, query: str) -> str:
        """
        Format the retrieved policy answer using a template.
        """
        try:
            answer = answer.strip().rstrip('.')
            query_lower = query.lower()
            if query_lower in self.greeting_words:
                template = self._select_template(query, 'greeting')
                return template
            
            policy_type = 'default'
            if 'customer satisfaction' in query_lower:
                policy_type = 'customer_satisfaction'
            elif 'it' in query_lower or 'technology' in query_lower:
                policy_type = 'it_policy'
            
            template = self._select_template(query, policy_type)
            if 'and' in answer:
                parts = [part.strip() for part in answer.split('and')]
                return template.format(*parts)
            return template.format(answer)
        except Exception as e:
            self.logger.warning(f"Answer formatting failed: {e}")
            return answer

    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the generative model, using the provided policy context.
        
        The prompt explicitly instructs the model to base the answer only on the provided text.
        """
        try:
            prompt = (
                "You are an organizational policy assistant. "
                "Use the following policy context to answer the query. Do not add any information not present in the context.\n\n"
                f"Policy Context: {context}\n\n"
                f"Query: {query}\n\nAnswer:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output = self.gen_model.generate(
                **inputs,
                max_length=550,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = generated_text.replace(prompt, "").strip()
            if not response:
                response = generated_text.strip()
            return response
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return "I encountered an error while generating a response."

    def get_answer(self, query: str, confidence_threshold: float = 0.7) -> str:
        """
        Retrieve the most relevant policy answer from the knowledge base or, if retrieval confidence is low,
        use the generative model with the best available context.
        """
        try:
            query_lower = query.lower().strip()
            # Quick response for greetings.
            if query_lower in self.greeting_words:
                return self._select_template(query, 'greeting')
            
            # Retrieve the best matching question from the index.
            query_embedding = self.retrieval_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            distances, indices = self.index.search(query_embedding, k=1)
            best_distance = distances[0][0]
            best_match = indices[0][0]
            retrieved_context = self.answers[self.questions[best_match]]
            
            # If the retrieval confidence is low, use the generative fallback with the retrieved context.
            if best_distance < confidence_threshold:
                self.logger.info("Low retrieval confidence; using generative model with retrieved context.")
                return self._generate_response(query, retrieved_context)
            
            # Otherwise, return the retrieved answer formatted via templating.
            return self._format_policy_answer(retrieved_context, query)
        
        except Exception as e:
            self.logger.error(f"Answer retrieval failed: {e}")
            return "I encountered an error while processing your query."

def configure_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = configure_logging()
    try:
        qa_system = PolicyQASystem(
            data_path=r"C:\Users\moksh\classroom\chatbot_deepseek\industry_chatbot\data\knowledge_base\cleaned_augmented_qa_pairs.json",
            logger=logger
        )
        
        # Example queries.
        queries = [
            "hi",
            "hey",
            "Explain network documentation policy according to the IT policy",
            "What is our customer satisfaction approach?",
            "Tell me about laptop maintenance",
            "What is our leave policy?"
        ]
        
        for query in queries:
            response = qa_system.get_answer(query)
            print(f"Query: {query}\nResponse: {response}\n")
    
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()
