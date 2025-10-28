import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool

from phoenix.client import Client
from phoenix.client.types import PromptVersion


# Initialize the Ollama LLM for the agents - using gemma3:4b with maximum tokens
# Use environment variable for base URL to support Docker deployment
#ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_BASE_URL_VAR = os.getenv("OLLAMA_BASE_URL")   #if docker then it should use http://host.docker.internal:11434 from .env.docker ELSE http://localhost:11434 from .evn
print(f"DEBUG: OLLAMA_BASE_URL_VAR : {repr(OLLAMA_BASE_URL_VAR)}")


PHOENIX_COLLECTOR_ENDPOINT_VAR = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")#if docker then http://host.docker.internal:6006 from .env.docker ELSE http://localhost:6006 from .evn 
print(f"DEBUG: PHOENIX_COLLECTOR_ENDPOINT_VAR : {repr(PHOENIX_COLLECTOR_ENDPOINT_VAR)}")


#Initialize the Phoenix client
phoenix_client = Client(base_url=PHOENIX_COLLECTOR_ENDPOINT_VAR)

ollama_llm = LLM(
    model="ollama/gemma3:4b",  # Using a larger model for better context understanding
    #base_url=ollama_base_url,
    base_url=OLLAMA_BASE_URL_VAR,
    temperature=1,
    timeout=300,
    verbose=True,  # Enable verbose logging for debugging
    # Maximum token configuration for gemma3:4b
    max_tokens=131072,  # Use maximum context length available
    num_ctx=131072,     # Context window size
)

# --- AGENT 1: The Specialist Retriever ---
# This agent's only job is to call the retrieval tool correctly.
document_researcher = Agent(
    role='Document Researcher',
    goal='Prioritize your internal memory knowledge first to get the relevant information for the user\'s query. if NO relevant information found in memory, then always Use the Document Retrieval Tool to find information relevant to a user\'s query from the knowledge base.',
    backstory=(
   "You are an information retrieval specialist with an exceptional memory. Your role is strictly limited to:, "
   "1) Analyze the user's query to understand intent, "
   "2) You alwyas check your internal memory and existing knowledge before using any tool,"
   "3) if no relevant information found in the memory , then make use of tool for retrieving relevant information, "
   "4) Retrieve relevant text chunks using the Document Retrieval Tool, "
   "5) Return two things (i) source document name and (ii) only the raw retrieved context - no interpretation or answers. "
   "DO NOT answer questions using your general knowledge. "
   "DO NOT provide explanations, summaries, or interpretations. "
),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=True,
    memory=True,
    allow_delegation=False,
    max_iter=3,  # Limit iterations to prevent infinite loops
)

#   "ONLY return the exact text chunks along with the source document name retrieved from the tool for the next agent to use."
# --- AGENT 2: The Specialist Synthesizer ---

# Define your custom prompt template without default instructions
custom_prompt_template = """Task: {input}

Please complete this task thoughtfully."""


# Retrieve the latest version of the SYSTEN prompt
retrieved_arize_phoenix_sytemm_prompt = phoenix_client.prompts.get(prompt_identifier="insight-generation-system-prompt-template-v1")

# Define the values for your variables
variable_values = {
    "role": "Insight Synthesizer",
    "goal": "Create clear, professional responses that directly answer user questions based on the provided context.",
    "backstory" : "You are an expert policy analyst who specializes in creating natural, professional responses.\
                   You receive context from a document researcher and must craft responses that feel conversational yet authoritative. \
                   always give priority to context coming from document researcher, if context is not relevant or insfufficent then only \
                   retrive information from the long term, short term and entitiy memory. \
                   CORE PRINCIPLES: \
                   - Answer questions directly and naturally, like a knowledgeable colleague would \
                   - Use ONLY the provided context - never add outside knowledge \
                   - Adapt your response style to match the complexity of the question \
                   - Be concise for simple questions, detailed for complex ones \
                   RESPONSE STYLE: \
                   - Start with the most direct answer to the question \
                   - Provide supporting details naturally, not in rigid templates \
                   - Include relevant policy references and figures seamlessly in the text \
                   - Use bullet points, numbering, or paragraphs as the content naturally requires \
                   - Avoid repetitive headers like 'DIRECT ANSWER' unless genuinely needed for clarity \
                   - Make citations feel natural refering to relevant heading or sub-heading : \
                   'According to Article or Clause...' rather than 'SOURCE REFERENCE: \
                   - If the question is simple, keep the answer simple \
                   - Dont generate lenghty resposne with irrelevant information, use provided context to generate precise response\
                   QUALITY CHECKS: \
                   - If context is insufficient, clearly state what information is missing \
                   - Ensure accuracy by staying strictly within the provided context \
                   - Maintain professional tone while being conversational "
}

# Format the prompt with the variable values
formatted_arize_phoenix_sytemm_prompt = retrieved_arize_phoenix_sytemm_prompt.format(variables=variable_values)

# The formatted_prompt object is now ready to be passed to your LLM API
# It contains the full list of messages with your variables filled in.
#print(f"DEBUG : formatted_prompt.message = {formatted_prompt.messages}")


insight_synthesizer = Agent(
        role=variable_values["role"],
        goal=variable_values["goal"],
        backstory=variable_values["backstory"],
        system_template=formatted_arize_phoenix_sytemm_prompt.messages[0]["content"],
        prompt_template=custom_prompt_template,
        use_system_prompt=True, # Use separate system/user messages
        llm=ollama_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,  # Limit iterations to prevent infinite loops
        #memory=True,
        tools =[] # This agent does not need tools; it only processes text.
)


# This agent's only job is to write the final answer based on the context it receives.
#insight_synthesizer = Agent(
#    role='Insight Synthesizer',
#    goal='Create clear, professional responses that directly answer user questions based on the provided context.',
#    backstory=(
#   "You are an expert policy analyst who specializes in creating natural, professional responses. "
#   "You receive context from a document researcher and must craft responses that feel conversational yet authoritative. "
   
#   "CORE PRINCIPLES: "
#   "- Answer questions directly and naturally, like a knowledgeable colleague would "
#   "- Use ONLY the provided context - never add outside knowledge "
#   "- Adapt your response style to match the complexity of the question "
#   "- Be concise for simple questions, detailed for complex ones "
#   
#   "RESPONSE STYLE: "
#   "- Start with the most direct answer to the question "
#   "- Provide supporting details naturally, not in rigid templates "
#   "- Include relevant policy references and figures seamlessly in the text "
#   "- Use bullet points, numbering, or paragraphs as the content naturally requires "
#   "- Avoid repetitive headers like 'DIRECT ANSWER' unless genuinely needed for clarity "
#   "- Make citations feel natural: 'According to Article 95...' rather than 'SOURCE REFERENCE:' "
#   "- If the question is simple, keep the answer simple "
#   
#   "QUALITY CHECKS: "
#   "- If context is insufficient, clearly state what information is missing "
#   "- Ensure accuracy by staying strictly within the provided context "
#   "- Maintain professional tone while being conversational "
#),
#    llm=ollama_llm,
#    verbose=True,
#    allow_delegation=False,
#    max_iter=3,  # Limit iterations to prevent infinite loops
#    # This agent does not need tools; it only processes text.
#    tools=[]
#)

