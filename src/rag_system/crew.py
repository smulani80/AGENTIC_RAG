import os 
import re
import traceback 

from crewai import Crew, Process, Task
from .agents import document_researcher, insight_synthesizer
from typing import Tuple, Any
from crewai import TaskOutput

from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage

from dotenv import load_dotenv



def check_for_confidential_info(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate content for sensitive information like UAE phone numbers."""
   
    uae_phone_numbers_pattern = r"\+971\s?[5-9]\d\s?\d{7}"
    uae_national_id_pattern = r"^784[ .-]?\d{4}[ .-]?\d{7}[ .-]?\d{1}$"

    try:
        content_text = str(result)
        
        # Debug: Log what we actually receive
        # print(f"DEBUG: Tool received content text as : {repr(content_text)}")

        #if re.search(uae_phone_numbers_pattern, content_text):
        #    return (False, "Confidential information (UAE_PHONE_NUMBER) detected. Content blocked.")
        #elif re.search(uae_national_id_pattern, content_text):
        #    return (False, "Confidential information (UAE_EMIRATES_ID) detected. Content blocked.")
        #else:
            # If no sensitive info is found, pass the content through
        #    return (True, result)


        status_msg = "OK"
		#msg_lst = []
		
        content_redacted = False
		
        # Debug: Log what we actually receive
        print(f"DEBUG: Tool received content text as : {repr(content_text)}")

        if re.search(uae_phone_numbers_pattern, content_text):
            masked_content_text1 = re.sub(uae_phone_numbers_pattern, "****PH.NO****", content_text)
            content_redacted = True
            msg1 = "Confidential information (UAE_PHONE_NUMBER) detected. Content redacted by masking it."
            print(f"DEBUG: {repr(msg1)}")
            #msg_lst.append(msg1)
            content_text = masked_content_text1
			
        if re.search(uae_national_id_pattern, content_text):
            masked_content_text2 = re.sub(uae_national_id_pattern, "****UAE.ID****", content_text)
            content_redacted = True
            msg2 = "Confidential information (UAE_EMIRATES_ID) detected. Content redacted by masking it."
            #msg_lst.append(msg2)
            content_text = masked_content_text2
			
        if content_redacted:
            #status_msg = " | ".join(msg_lst)
            return (False, content_text)	
        else:
            # If no sensitive info is found, pass the content through
            return (True, result)
    except Exception as e:
        traceback.print_exc()
        return (False, "Unexpected error during validation")


def create_rag_crew(query: str):

    """
    Creates a CrewAI instance with enhanced memory capabilities:
    1. Long-term Memory: Persistent storage using SQLite
    2. Short-term Memory: RAG-based memory for recent context
    3. Entity Memory: Tracks and maintains information about specific entities
    """
    # Load environment variables from a .env file
    load_dotenv()

    # Define the data directory for memory storage
    os.environ["CREWAI_STORAGE_DIR"] = "/home/ec2-user/nbs-agentic-rag/CREW_AI_MEM_STORE/"
    os.environ["DATA_DIR"] = "/home/ec2-user/nbs-agentic-rag/CREW_AI_MEM_STORE/DATA/db"

    DATA_DIR = os.getenv("DATA_DIR")
    OLLAMA_BASE_URL_VAR = os.getenv("OLLAMA_BASE_URL")   #if docker then it should use http://host.docker.internal:11434 from .env.docker ELSE http://localhost:11434 from .evn

    ollama_embedder_config = {
            "provider": "ollama",
            "config":{
                "model_name": "nomic-embed-text:v1.5",
                #"url": "http://localhost:11434" # Optional: Specify if Ollama is not running on default URL
                #"url": "http://host.docker.internal:11434" # Optional: Specify if Ollama is not running on default URL
                "url": OLLAMA_BASE_URL_VAR # Optional: Specify if Ollama is not running on default URL
            }
    }

    print(f"DEBUG: OLLAMA_BASE_URL_VAR : {repr(OLLAMA_BASE_URL_VAR)}")

    # Initialize memory components
    long_term_memory = LongTermMemory(
        storage=LTMSQLiteStorage(db_path=f"{DATA_DIR}/long_term_memory.db")
    )

    short_term_memory = ShortTermMemory(
        storage=RAGStorage(
            embedder_config=ollama_embedder_config,
            path=f"{DATA_DIR}/short_term_memory.db",
            type="short_term"
        )
    )

    entity_memory = EntityMemory(
        storage=RAGStorage(
            embedder_config=ollama_embedder_config,
            path=f"{DATA_DIR}/entity_memory.db",
            type="entities"
        )
    )


    """
    Creates and configures a three-agent RAG crew to process a query.
    - The Document Researcher finds relevant information.
    - The Insight Synthesizer formulates the final answer based on the retrieved context.
    - The Redactor Guardrail the final answer received from Synthesizer by excluding Personally Identifiable Information (PII like Name, Passport Number etc).
    """

    # Task for the Document Researcher agent
    # This task focuses exclusively on using the tool to find information.
    research_task = Task(
            description=f"First try to fetch highly contextually similar or exact information from the memory for query: '{query}',Otherwise Always find relevant information in the documents for the query: '{query}'.",
        expected_output="A block of text containing chunks of the most relevant document sections and respective source document file names.",
        agent=document_researcher
    )

    # Task for the Insight Synthesizer agent
    # This task takes the context from the first task and focuses on crafting the answer.
    synthesis_task = Task(
        description=f"Analyze the provided document context from {research_task} and formulate a comprehensive and accurate answer to the user's original question: '{query}'.",
        expected_output="""A professional, well-structured response that directly answers the user's question. Format the response naturally and appropriately based on the content:

Guidelines for response formatting:
- Start with a clear, direct answer to the question
- Provide supporting details, explanations, or calculations only when relevant
- Include specific references to policy articles, sections, or documents when citing sources
- Use natural language flow rather than rigid templates
- Adapt the structure to fit the content (simple answers for simple questions, detailed breakdowns for complex ones)
- Use proper formatting (bullet points, numbering, or paragraphs) as appropriate for the content
- Ensure professional tone and clarity
- Include precise figures, timeframes, and regulatory references where applicable
- Always make sure source document information like name,page number is mentioend in the generated answer
- Don't generate lenghty response with irrelevant information

The response should feel conversational yet authoritative, avoiding repetitive headers unless the content genuinely requires structured breakdown.""",
        agent=insight_synthesizer,
        context=[research_task], # This ensures it uses the output from the research_task
        guardrail=check_for_confidential_info, # Task level guardrail function.
        guardrail_max_retries=3, # Limit retry attempts
    )


    # Create the crew with a sequential process
    rag_crew = Crew(
        agents=[document_researcher, insight_synthesizer],
        tasks=[research_task, synthesis_task],
        process=Process.sequential, # The tasks will be executed one after the other
        embedder=ollama_embedder_config,
        memory=True,
        cache=True,
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        entity_memory=entity_memory,
        verbose=True,
    )

    return rag_crew
