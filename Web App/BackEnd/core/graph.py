import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.models import GraphState
from core.agents import (
    validator_agent,
    resume_parser_agent,
    jd_parser_agent,
    ats_score_before_node,
    gap_analysis_agent,
    resume_enhancement_agent,
    critic_agent,
    ats_score_after_node,
    report_generator_agent,
    routing_logic,
    reflection_routing,
)

logger = logging.getLogger("ai_resume_optimizer.graph")


def build_resume_analyzer_graph(with_checkpointer: bool = True):
    """
    Construct and compile the LangGraph workflow with parallel parsing,
    self-reflection, and optional MemorySaver checkpointing.
    """
    workflow = StateGraph(GraphState)
    
    # 1. Register nodes
    workflow.add_node("validator", validator_agent)
    workflow.add_node("resume_parser", resume_parser_agent)
    workflow.add_node("jd_parser", jd_parser_agent)
    workflow.add_node("ats_score_before", ats_score_before_node)
    workflow.add_node("gap_analysis", gap_analysis_agent)
    workflow.add_node("resume_enhancement", resume_enhancement_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("ats_score_after", ats_score_after_node)
    workflow.add_node("report_generator", report_generator_agent)
    
    # 2. Guardrail check
    workflow.add_edge(START, "validator")
    workflow.add_conditional_edges(
        "validator",
        routing_logic,
        {"invalid": END, "valid": "resume_parser"}
    )
    
    # 3. Parallel parsing fan-out
    workflow.add_edge("validator", "jd_parser")
    
    # 4. Fan-in to ATS scoring before
    workflow.add_edge("resume_parser", "ats_score_before")
    workflow.add_edge("jd_parser", "ats_score_before")
    
    # 5. Gap analysis & enhancement
    workflow.add_edge("ats_score_before", "gap_analysis")
    workflow.add_edge("gap_analysis", "resume_enhancement")
    
    # 6. Self-reflection loop
    workflow.add_edge("resume_enhancement", "critic")
    workflow.add_conditional_edges(
        "critic",
        reflection_routing,
        {"fail": "resume_enhancement", "pass": "ats_score_after"}
    )
    
    # 7. Final report generation
    workflow.add_edge("ats_score_after", "report_generator")
    workflow.add_edge("report_generator", END)
    
    checkpointer = MemorySaver() if with_checkpointer else None
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("LangGraph workflow compiled successfully.")
    return compiled_graph
