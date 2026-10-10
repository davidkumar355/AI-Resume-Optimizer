import asyncio
import json
import time
import logging
from typing import AsyncGenerator, Optional, Dict, Any

from core.graph import build_resume_analyzer_graph

logger = logging.getLogger("ai_resume_optimizer.pipeline_service")

# Progress percentages and friendly labels for each agent
AGENT_METADATA = {
    "validator": {
        "label": "Document Validator",
        "message": "Validating uploaded document structure...",
        "progress": 10
    },
    "resume_parser": {
        "label": "Resume Parser",
        "message": "Extracting structured skills, experience, and education...",
        "progress": 25
    },
    "jd_parser": {
        "label": "Job Description Parser",
        "message": "Analyzing required qualifications and responsibilities...",
        "progress": 35
    },
    "ats_score_before": {
        "label": "Baseline ATS Scorer",
        "message": "Evaluating pre-optimization ATS score across 5 dimensions...",
        "progress": 50
    },
    "gap_analysis": {
        "label": "Gap Analysis Engine",
        "message": "Identifying missing skills and keyword opportunities...",
        "progress": 60
    },
    "resume_enhancement": {
        "label": "Resume Enhancement Writer",
        "message": "Rewriting bullet points with STAR/XYZ metrics and keywords...",
        "progress": 75
    },
    "critic": {
        "label": "Critic Agent",
        "message": "Reviewing enhancements for hallucination and keyword stuffing...",
        "progress": 85
    },
    "ats_score_after": {
        "label": "Optimized ATS Scorer",
        "message": "Recalculating ATS score with enhanced bullets...",
        "progress": 92
    },
    "report_generator": {
        "label": "Report Generator",
        "message": "Compiling final comparative recommendations...",
        "progress": 98
    },
}


def format_sse(event: str, data: Dict[str, Any]) -> str:
    """Formats an event and dictionary into standard SSE string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_pipeline(
    resume_text: str,
    jd_text: str,
    api_key: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Executes the LangGraph optimization workflow asynchronously in a background thread,
    yielding real-time Server-Sent Events (SSE) to the client.
    """
    start_time = time.time()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Accumulated state throughout the run
    accumulated_state: Dict[str, Any] = {
        "resume_text": resume_text,
        "jd_text": jd_text,
    }

    def sync_worker():
        try:
            initial_state = {
                "resume_text": resume_text,
                "jd_text": jd_text,
                "api_key": api_key or "",
            }

            app = build_resume_analyzer_graph(with_checkpointer=False)

            # Send initial event
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("status", {
                    "agent": "system",
                    "label": "System",
                    "message": "Pipeline initialized. Beginning validation...",
                    "progress": 5
                })
            )

            # Stream LangGraph node executions
            for chunk in app.stream(initial_state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    accumulated_state.update(node_output)
                    meta = AGENT_METADATA.get(node_name, {
                        "label": node_name.replace("_", " ").title(),
                        "message": f"Completed {node_name}...",
                        "progress": 50
                    })

                    # Check for invalid document abort
                    if node_name == "validator" and not node_output.get("is_valid_resume", True):
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("error", {
                                "message": "The uploaded document does not appear to be a valid resume.",
                                "code": "INVALID_DOCUMENT"
                            })
                        )
                        return

                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("status", {
                            "agent": node_name,
                            "label": meta["label"],
                            "message": meta["message"],
                            "progress": meta["progress"],
                            "data_preview": {
                                k: v for k, v in node_output.items() if k not in ("resume_text", "jd_text")
                            }
                        })
                    )

            # All nodes completed successfully -> push final result
            elapsed = round(time.time() - start_time, 1)
            result_payload = {
                "resume_data": accumulated_state.get("resume_data", {}),
                "jd_data": accumulated_state.get("jd_data", {}),
                "ats_score_before": accumulated_state.get("ats_score_before", {}),
                "gap_analysis": accumulated_state.get("gap_analysis", {}),
                "optimized_resume": accumulated_state.get("optimized_resume", {}),
                "ats_score_after": accumulated_state.get("ats_score_after", {}),
                "final_report": accumulated_state.get("final_report", {}),
            }

            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("result", result_payload)
            )

            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("done", {
                    "message": "Resume optimization completed successfully!",
                    "elapsed_seconds": elapsed
                })
            )

        except Exception as exc:
            logger.exception("Error during pipeline execution")
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("error", {
                    "message": str(exc),
                    "code": "PIPELINE_ERROR"
                })
            )
        finally:
            # Sentinel to signal end of stream
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Launch background thread for LangGraph execution
    worker_task = asyncio.to_thread(sync_worker)
    asyncio.create_task(worker_task)

    # Yield SSE events from queue as they arrive
    while True:
        item = await queue.get()
        if item is None:
            break
        event_name, data = item
        yield format_sse(event_name, data)
