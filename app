
from langgraph.graph import END, START, StateGraph

# from ipython import IPyton

# from IPython.display import Image, display


from schemas.rag_state import RAGState

from helperfuncions.classify_query import classifier_node
from helperfuncions.classify_input_type import classify_ticket_type_node
from helperfuncions.keyword_extractor import extract_keywords_node
from helperfuncions.get_docs_node import get_docs_node
from helperfuncions.simple_answer import answer_simple_node
from helperfuncions.grade_docs import grade_docs_node
from helperfuncions.answer_rag import answer_rag_node
from helperfuncions.reformulate_node import reformulate_query_node


# grade_docs_prompt=PROMPT_LIBRARY["grader_prompt"]
# grade_docs_prompt_template=PromptTemplate(input_variables=['query','docs'],template=grade_docs_prompt)
# grade_docs_chain= grade_docs_prompt_template | openai_llm_o1.with_structured_output(GradeOutput)

# answer_docs_prompt=PROMPT_LIBRARY["answer_prompt"]
# answer_docs_prompt_template=PromptTemplate(input_variables=['query','docs'],template=answer_docs_prompt)
# answer_docs_chain = answer_docs_prompt_template | openai_llm_o1 | StrOutputParser()

# reformulate_query_prompt=PROMPT_LIBRARY["reformuate_query"]
# reformulate_query_prompt_template=PromptTemplate(input_variables=['query','docs'],template=reformulate_query_prompt)
# reformulate_query_chain=reformulate_query_prompt_template | openai_llm_o1 | StrOutputParser() 

def rag_builder_app(user_input: str):
    state: RAGState = {
        "query": user_input,
        "retries": 0,
        "max_retries": 2,
    }

    builder = StateGraph(RAGState)

    # Register all nodes
    builder.add_node("classify", classifier_node)
    builder.add_node("classify_ticket_type", classify_ticket_type_node)
    builder.add_node("extract_keywords", extract_keywords_node)
    builder.add_node("get_docs", get_docs_node)
    builder.add_node("grade_docs", grade_docs_node)
    builder.add_node("reformulate_query", reformulate_query_node)
    builder.add_node("answer_simple", answer_simple_node)
    builder.add_node("answer_rag", answer_rag_node)

    # Classify → route
    
    #Set Entry Point
    builder.set_entry_point("classify")
    builder.add_edge(START,"classify")

    def classify_router(state: RAGState) -> str:
        return "classify_ticket_type" if state["input_type"] == "rag" else "answer_simple"
    
    builder.add_conditional_edges("classify", classify_router, {
        "classify_ticket_type": "classify_ticket_type",
        "answer_simple": "answer_simple"
    })

    
 #   builder.add_edge("classify", "classify_ticket_type")

    builder.add_edge("classify_ticket_type", "extract_keywords")
    builder.add_edge("extract_keywords", "get_docs")
    
    # Docs → Grade
    #builder.add_edge("classify", "get_docs")
    builder.add_edge("get_docs", "grade_docs")

    # Grade → Answer or Reformulate
    def grade_router(state: RAGState) -> str:
        return "answer_rag" if state["grade"] == "Relevant" else "reformulate_query"

    builder.add_conditional_edges("grade_docs", grade_router, {
        "answer_rag": "answer_rag",
        "reformulate_query": "reformulate_query"
    })

    # Reformulate → Retry or Final Answer
    def retry_guard(state: RAGState) -> str:
        if state["retries"] >= state.get("max_retries", 2):
            return "answer_rag"
        return "extract_keywords"

    builder.add_conditional_edges("reformulate_query", retry_guard, {
        "extract_keywords": "extract_keywords",
        "answer_rag": "answer_rag"
    })

    # Terminal nodes
    builder.add_edge("answer_simple", END)
    builder.add_edge("answer_rag", END)

    # Compile & Run
    graph = builder.compile()
  #  display(Image(graph.get_graph().draw_mermaid_png()))
    result = graph.invoke(state)
    print("🧠 Final State:", result)
    return result
