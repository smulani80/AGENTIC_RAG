import os
import sys
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv

from llama_index.core import Settings                              #Added to avoid OPEN_API_KEY error due to default OpenAI model fallback 
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
#from ragas.integrations.llama_index import LlamaIndexLLM as RagasLlamaIndexLLM
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
                                                                   #Added to avoid OPEN_API_KEY error due to default OpenAI model fallback 

# Add the project root to the Python path to allow importing from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Now you can import from your modules
from src.rag_system.crew import create_rag_crew

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Specify the path to your golden dataset
EVAL_DATASET_PATH = os.path.join(project_root, 'src', 'evaluation', 'eval_dataset.jsonl')

def run_rag_pipeline(query: str):
    """
    A wrapper function to run the CrewAI RAG pipeline and return the final result.
    """
    try:
        rag_crew = create_rag_crew(query)
        result = rag_crew.kickoff()
        
        # --- FIX: Convert the CrewAI output object to a plain string ---
        # The 'datasets' library expects simple data types like strings, not complex objects.
        # Casting the result to a string extracts the final answer.
        answer_string = str(result)
        
        # For this evaluation, we treat the entire final output as the 'answer'.
        # The 'contexts' for RAGAS are the pieces of text the answer is based on.
        # Since the agent's final answer is a synthesis of the retrieved context,
        # we will use the answer itself as the context to measure faithfulness.
        answer = answer_string
        contexts = [answer_string] # Use the string version here as well

        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        print(f"Error running crew for query '{query}': {e}")
        return {"answer": "Error", "contexts": []}

async def main():
    """
    Main function to run the RAGAS evaluation.
    """
    print(f"📚 Loading evaluation dataset from: {EVAL_DATASET_PATH}")
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"❌ Error: Evaluation dataset not found at {EVAL_DATASET_PATH}")
        return
                                                                   #Added to avoid OPEN_API_KEY error due to default OpenAI model fallback

    # Initialize the embedding model - use environment variable for base URL to support Docker 
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5", 
            base_url=ollama_base_url,
            request_timeout=120.0  # Increased timeout for slower connections
            )

    # Configure the local LLM to prevent fallback to OpenAI
    # Be sure you have a local LLM model running, e.g., 'llama3' or 'gemma3:4b'
    ollama_llm = Ollama(model="gemma3:4b",
                        base_url=ollama_base_url,
                        request_timeout=120.0  # Increased timeout for slower connections
                        )

    # Set the embedding model in LlamaIndex's global settings
    # Set the global defaults for both the LLM and the embedding model
    Settings.llm = ollama_llm
    Settings.embed_model = embed_model

    # Wrap the Ollama LLM instance in the Ragas LlamaIndex wrapper.
    #ragas_ollama_llm = RagasLlamaIndexLLM(llama_index_llm=ollama_llm)
    #ragas_ollama_llm = LlamaIndexLLMWrapper(llama_index_llm=ollama_llm)
    ragas_ollama_llm = LlamaIndexLLMWrapper(llm=ollama_llm)
    ragas_ollama_embed_model = LlamaIndexEmbeddingsWrapper(embeddings=embed_model)

    # Assign Ollama models to Ragas metrics

    # Set the wrapped LLM for the specific Ragas faithfulness metric.
    # This overrides the default LLM used by the faithfulness evaluation.

    faithfulness.llm = ragas_ollama_llm
    answer_relevancy.llm = ragas_ollama_llm
    context_precision.llm = ragas_ollama_llm
    context_recall.llm = ragas_ollama_llm
    answer_relevancy.embeddings = ragas_ollama_embed_model
                                                                   #Added to avoid OPEN_API_KEY error due to default OpenAI model fallback


    # Load the golden dataset from the .jsonl file
    golden_dataset = Dataset.from_json(EVAL_DATASET_PATH)

    # --- Run the RAG pipeline for each question in the dataset ---
    print("\n🚀 Running RAG pipeline on the evaluation dataset...")
    results = []
    questions = []
    ground_truths = []
    
    for entry in golden_dataset:
        question = entry['question']
        ground_truth = entry['ground_truth']
        
        print(f"  - Processing question: '{question[:80]}...'")
        pipeline_output = run_rag_pipeline(question)
        
        results.append(pipeline_output)
        questions.append(question)
        ground_truths.append(ground_truth)

    # --- Prepare the dataset for RAGAS evaluation ---
    evaluation_data = {
        "question": questions,
        "answer": [res["answer"] for res in results],
        "contexts": [res["contexts"] for res in results],
        "ground_truth": ground_truths,
    }
    eval_dataset = Dataset.from_dict(evaluation_data)

    # --- Run the RAGAS evaluation ---
    print("\n📊 Evaluating the results with RAGAS...")
    
    # Define the metrics we want to use
    metrics = [
        faithfulness,       # How factually accurate is the answer based on the context?
        answer_relevancy,   # How relevant is the answer to the question?
        context_recall,     # Did the retriever find all the relevant context?
        context_precision,  # Was the retrieved context precise and not full of noise?
    ]

    # Run the evaluation it return an object of type EvaluationResult 
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_ollama_llm # Pass the Ragas-wrapped Ollama LLM to the evaluation
    )

    print("\n🎉 Evaluation Complete!")
    print("-------------------------")
    print(result)
    print("-------------------------")

    #EvaluationResult object has a convenient to_pandas() method that converts the results into a DataFrame
    result_df = result.to_pandas()
    result_df.to_csv('run_ragas_eval_output.csv', index=False)

    print("\n🎉 Evaluation Report has been saved as - run_ragas_eval_output.csv !")
    print("-------------------------")
    print(result_df)
    print("-------------------------")



if __name__ == "__main__":
    # Ragas evaluation uses asyncio, so we run the main function in an event loop
    asyncio.run(main())
