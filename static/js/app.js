const runForm = document.getElementById("runForm");
const topicInput = document.getElementById("topicInput");
const runButton = document.getElementById("runButton");
const cancelButton = document.getElementById("cancelButton");
const newRunButton = document.getElementById("newRunButton");

const timeline = document.getElementById("timeline");
const activityEmpty = document.getElementById("activityEmpty");

const planList = document.getElementById("planList");
const planEmpty = document.getElementById("planEmpty");
const planMeta = document.getElementById("planMeta");
const taskCount = document.getElementById("taskCount");

const progressRing = document.getElementById("progressRing");
const progressText = document.getElementById("progressText");
const runBadge = document.getElementById("runBadge");

const articlePreview = document.getElementById("articlePreview");
const markdownOutput = document.getElementById("markdownOutput");

const previewTab = document.getElementById("previewTab");
const markdownTab = document.getElementById("markdownTab");

const copyButton = document.getElementById("copyButton");
const downloadButton = document.getElementById("downloadButton");
const downloadZipButton = document.getElementById("downloadZipButton");

const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");
const toast = document.getElementById("toast");

let abortController = null;
let finalMarkdown = "";
let totalTasks = 0;
let completedTasks = 0;

let currentProgress = 0;

const stageProgress = {
    input_guardrails: 5,
    router: 10,
    research: 15,
    planner: 20,
    writing: 40,
    editor: 60,
    output_guardrails: 70,
    image_planner: 85,
    image_generator: 92,
    publication_qa: 98,
    completed: 100,
};


function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");

    window.setTimeout(() => {
        toast.classList.remove("show");
    }, 2200);
}


function updateProgress(value) {
    const val = Math.max(0, Math.min(100, Math.round(value)));
    if (val >= currentProgress) {
        currentProgress = val;
        if (progressRing) {
            progressRing.style.setProperty("--progress", currentProgress);
        }
        if (progressText) {
            progressText.textContent = `${currentProgress}%`;
        }
    }
}


function resetProgress() {
    currentProgress = 0;
    if (progressRing) {
        progressRing.style.setProperty("--progress", 0);
    }
    if (progressText) {
        progressText.textContent = "0%";
    }
}


function resetInterface() {
    finalMarkdown = "";
    totalTasks = 0;
    completedTasks = 0;

    timeline.innerHTML = "";
    planList.innerHTML = "";

    activityEmpty.hidden = false;
    planEmpty.hidden = false;

    timeline.appendChild(activityEmpty);
    planList.appendChild(planEmpty);

    planMeta.innerHTML = "";
    planMeta.hidden = true;

    taskCount.textContent = "0 tasks";

    articlePreview.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">✎</div>
            <h3>Generated Markdown will appear here</h3>
            <p>
                Generated Markdown will appear here after the
                workflow completes.
            </p>
        </div>
    `;

    markdownOutput.textContent = "";
    markdownOutput.hidden = true;
    articlePreview.hidden = false;

    previewTab.classList.add("active");
    markdownTab.classList.remove("active");

    copyButton.disabled = true;

    downloadButton.href = "#";
    downloadButton.classList.add("disabled");
    downloadButton.setAttribute(
        "aria-disabled",
        "true",
    );

    if (downloadZipButton) {
        downloadZipButton.href = "#";
        downloadZipButton.classList.add("disabled");
        downloadZipButton.setAttribute(
            "aria-disabled",
            "true",
        );
    }

    runBadge.textContent = "Ready";
    runBadge.className = "run-badge";

    resetProgress();
}


function setRunningState(isRunning) {
    topicInput.disabled = isRunning;
    runButton.disabled = isRunning;

    cancelButton.hidden = !isRunning;

    if (isRunning) {
        runBadge.textContent = "Running";
        runBadge.className = "run-badge running";
    }
}


const ORCHESTRATION_STAGES = [
    { id: "input_guardrails", label: "Input Guardrails" },
    { id: "router", label: "Route Request" },
    { id: "research", label: "Retrieve Evidence" },
    { id: "planner", label: "Build Article Plan" },
    { id: "writing", label: "Generate Sections" },
    { id: "editor", label: "Editorial Review" },
    { id: "output_guardrails", label: "Output Guardrails" },
    { id: "image_planner", label: "Image Planner" },
    { id: "image_generator", label: "Generate Images" },
    { id: "publication_qa", label: "Publication QA" },
    { id: "completed", label: "Completed" },
];

function initializeTimeline() {
    if (!timeline) return;

    activityEmpty.hidden = true;
    timeline.hidden = false;
    timeline.innerHTML = "";

    ORCHESTRATION_STAGES.forEach(stage => {
        const item = document.createElement("div");
        item.className = "timeline-stage waiting";
        item.dataset.stageId = stage.id;
        item.innerHTML = `
            <div class="stage-status-icon">⚪</div>
            <div class="stage-info">
                <div class="stage-title">${escapeHtml(stage.label)}</div>
                <div class="stage-detail">Waiting</div>
            </div>
        `;
        timeline.appendChild(item);
    });
}

function updateStageStatus(stageId, status, detail = null) {
    if (!stageId) return;

    activityEmpty.hidden = true;
    if (timeline.hidden) timeline.hidden = false;
    if (timeline.children.length === 0) initializeTimeline();

    const stageMap = {
        input_guardrails: "input_guardrails",

        router: "router",
        routing: "router",

        research: "research",
        tavily_worker: "research",
        research_complete: "research",

        planner: "planner",
        orchestrator: "planner",
        plan: "planner",

        writing: "writing",
        workers: "writing",
        worker_section: "writing",
        assemble_sections: "writing",
        section_complete: "writing",

        editor: "editor",

        output_guardrails: "output_guardrails",

        formatter: "editor",
        markdown_formatter: "editor",

        image_planner: "image_planner",
        decide_images: "image_planner",
        images_planned: "image_planner",

        image_generator: "image_generator",
        image_worker: "image_generator",
        publishing: "image_generator",
        generate_images: "image_generator",
        generate_and_place_images: "image_generator",

        completed: "completed",
        done: "completed"
    };

    const targetId = stageMap[stageId] || stageId;
    const stageEl = timeline.querySelector(`[data-stage-id="${targetId}"]`);
    if (!stageEl) return;

    const iconMap = {
        waiting: "⚪",
        running: "🟡",
        completed: "🟢",
        failed: "🔴"
    };

    stageEl.className = `timeline-stage ${status}`;
    const iconEl = stageEl.querySelector(".stage-status-icon");
    const detailEl = stageEl.querySelector(".stage-detail");

    if (iconEl) iconEl.textContent = iconMap[status] || "⚪";
    if (detailEl) {
        if (detail) {
            detailEl.textContent = detail;
        } else if (status === "running") {
            detailEl.textContent = "Running...";
        } else if (status === "completed") {
            detailEl.textContent = "Completed";
        } else if (status === "failed") {
            detailEl.textContent = "Failed";
        }
    }

    if ((status === "running" || status === "completed") && stageProgress[targetId] !== undefined) {
        updateProgress(stageProgress[targetId]);
    }
}

function updateStage(event) {
    updateStageStatus(event.id, event.status || "running", event.detail);
}

function addSubstage(event) {
    updateStageStatus(event.id || event.label, event.status || "completed", event.detail);
}

function showRouting(event) {
    const modeLabels = {
        closed_book: "Selected closed book mode",
        hybrid: "Selected hybrid mode",
        open_book: "Selected open book mode",
    };
    const detail = `${modeLabels[event.mode] || event.mode}. Research ${event.needs_research ? "required" : "not required"}.`;
    updateStageStatus("router", "completed", detail);
}

function showResearch(event) {
    const count = event.evidence ? event.evidence.length : 0;
    updateStageStatus("research", "completed", `Prepared ${count} deduplicated evidence sources.`);
}


function showPlan(plan) {
    planEmpty.hidden = true;
    planList.innerHTML = "";

    const tasks = plan.tasks || [];

    totalTasks = tasks.length;
    completedTasks = 0;

    taskCount.textContent =
        `${tasks.length} ${tasks.length === 1 ? "task" : "tasks"
        }`;

    planMeta.hidden = false;

    planMeta.innerHTML = `
        <div>
            <span>Audience</span>
            <strong>
                ${escapeHtml(plan.audience || "Developers")}
            </strong>
        </div>

        <div>
            <span>Format</span>
            <strong>
                ${escapeHtml(
        (plan.blog_kind || "explainer")
            .replaceAll("_", " "),
    )}
            </strong>
        </div>

        <div>
            <span>Tone</span>
            <strong>
                ${escapeHtml(plan.tone || "Technical")}
            </strong>
        </div>
    `;

    tasks.forEach((task, index) => {
        const item = document.createElement("article");

        item.className = "plan-item";
        item.dataset.taskId = String(task.id);

        const tags = [];

        if (task.requires_research) {
            tags.push("Research");
        }

        if (task.requires_citations) {
            tags.push("Citations");
        }

        if (task.requires_code) {
            tags.push("Code");
        }

        const targetWords = task.target_words || task.target_word_count || task.word_count;
        const wordBadgeHtml = (targetWords !== undefined && targetWords !== null && targetWords !== "" && !Number.isNaN(targetWords))
            ? `<span>${escapeHtml(targetWords)} words</span>`
            : '';

        item.innerHTML = `
            <div class="plan-number">
                ${String(index + 1).padStart(2, "0")}
            </div>

            <div class="plan-content">
                <div class="plan-title-row">
                    <h3>
                        ${escapeHtml(task.title)}
                    </h3>

                    <span class="plan-status">
                        Waiting
                    </span>
                </div>

                <p>
                    ${escapeHtml(task.goal)}
                </p>

                <div class="plan-tags">
                    ${wordBadgeHtml}

                    ${tags.map(
            tag => `
                            <span>
                                ${escapeHtml(tag)}
                            </span>
                        `,
        ).join("")}
                </div>

                <div class="task-progress">
                    <span></span>
                </div>
            </div>
        `;

        planList.appendChild(item);
    });
}


function completeSection(event) {
    const taskElement = document.querySelector(
        `[data-task-id="${event.task_id}"]`,
    );

    if (taskElement) {
        taskElement.classList.add("completed");

        const status = taskElement.querySelector(
            ".plan-status",
        );

        status.textContent = "Completed";

        const progressBar = taskElement.querySelector(
            ".task-progress span",
        );

        progressBar.style.width = "100%";
    }

    completedTasks = event.completed || (
        completedTasks + 1
    );

    totalTasks = event.total || totalTasks;

    const startP = stageProgress.planner || 20;
    const endP = stageProgress.writing || 40;
    const workerProgress = totalTasks
        ? startP + (completedTasks / totalTasks) * (endP - startP)
        : endP;

    updateProgress(workerProgress);

    const workerStage = getStageElement("workers");

    if (workerStage) {
        const detail = workerStage.querySelector(
            ".timeline-detail",
        );

        detail.textContent =
            `${completedTasks} of ${totalTasks} sections completed.`;
    }
}


function displayFinalResult(event) {
    finalMarkdown = event.markdown || "";

    markdownOutput.textContent = finalMarkdown;

    const rendered = marked.parse(finalMarkdown);

    articlePreview.innerHTML = DOMPurify.sanitize(
        rendered,
    );

    copyButton.disabled = false;

    downloadButton.href = `${event.download_url}?format=md`;
    downloadButton.classList.remove("disabled");
    downloadButton.setAttribute(
        "aria-disabled",
        "false",
    );

    if (downloadZipButton) {
        downloadZipButton.href = `${event.download_url}?format=zip`;
        downloadZipButton.classList.remove("disabled");
        downloadZipButton.setAttribute(
            "aria-disabled",
            "false",
        );
    }

    runBadge.textContent = "Completed";
    runBadge.className = "run-badge completed";

    updateProgress(100);

    ORCHESTRATION_STAGES.forEach(stage => {
        const stageEl = timeline.querySelector(`[data-stage-id="${stage.id}"]`);
        if (stageEl && !stageEl.classList.contains("completed")) {
            updateStageStatus(stage.id, "completed", "Completed");
        }
    });

    document
        .getElementById("resultCard")
        .scrollIntoView({
            behavior: "smooth",
            block: "start",
        });
}


function displayError(message) {
    runBadge.textContent = "Failed";
    runBadge.className = "run-badge failed";

    const errorItem = document.createElement("div");

    errorItem.className =
        "timeline-item failed";

    errorItem.innerHTML = `
        <div class="timeline-marker">
            <span></span>
        </div>

        <div class="timeline-content">
            <div class="timeline-title">
                Workflow execution failed
            </div>

            <div class="timeline-detail">
                ${escapeHtml(message)}
            </div>
        </div>
    `;

    timeline.appendChild(errorItem);

    showToast("Workflow execution failed");
}


function handleEvent(event) {
    switch (event.type) {
        case "run_started":
            runBadge.textContent = "Running";
            break;

        case "stage":
            updateStage(event);
            break;

        case "substage":
            addSubstage(event);
            break;

        case "routing":
            showRouting(event);
            break;

        case "research_complete":
            showResearch(event);
            break;

        case "plan":
            showPlan(event.plan || {});
            break;

        case "summary":
            renderExecutionSummary(event.summary || {}, event.metrics || []);
            break;


        case "section_complete":
            completeSection(event);
            break;

        case "images_planned":
            showToast(
                `${event.count} visual${event.count === 1 ? "" : "s"
                } planned`,
            );
            break;

        case "qa_complete":
            if (event.status === "FAIL") {
                const fails = event.qa_result?.failures || [];
                showToast(`QA Failed: ${fails[0] || "Requirements not met"}`);
            } else {
                showToast("Publication QA Passed");
            }
            break;

        case "final":
            displayFinalResult(event);
            break;

        case "error":
            displayError(event.message);
            break;

        case "done":
            setRunningState(false);
            break;
    }
}


function parseSSEChunk(buffer) {
    const events = buffer.split("\n\n");
    const remaining = events.pop() || "";

    events.forEach(block => {
        const dataLines = block
            .split("\n")
            .filter(line => line.startsWith("data:"))
            .map(line => line.slice(5).trim());

        if (!dataLines.length) {
            return;
        }

        const rawData = dataLines.join("\n");

        try {
            const event = JSON.parse(rawData);
            handleEvent(event);
        } catch (error) {
            console.error(
                "Could not parse stream event:",
                rawData,
                error,
            );
        }
    });

    return remaining;
}


async function executeAgent(topic) {
    abortController = new AbortController();

    const lengthSelect = document.getElementById("lengthSelect");
    const targetWordCount = lengthSelect ? parseInt(lengthSelect.value, 10) : 3500;

    const response = await fetch("/api/run", {
        method: "POST",

        headers: {
            "Content-Type": "application/json",
        },

        body: JSON.stringify({
            topic,
            target_word_count: targetWordCount,
        }),

        signal: abortController.signal,
    });

    if (!response.ok) {
        const message = await response.text();

        throw new Error(
            message || "Could not start the agent.",
        );
    }

    if (!response.body) {
        throw new Error(
            "Streaming is not supported by this browser.",
        );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";

    while (true) {
        const {
            done,
            value,
        } = await reader.read();

        if (done) {
            break;
        }

        buffer += decoder.decode(
            value,
            {
                stream: true,
            },
        );

        buffer = parseSSEChunk(buffer);
    }

    buffer += decoder.decode();

    if (buffer.trim()) {
        parseSSEChunk(`${buffer}\n\n`);
    }
}


runForm.addEventListener(
    "submit",
    async event => {
        event.preventDefault();

        const topic = topicInput.value.trim();

        if (topic.length < 3) {
            showToast(
                "Enter a more detailed topic.",
            );
            return;
        }

        resetInterface();
        initializeTimeline();
        setRunningState(true);
        planEmpty.hidden = false;

        try {
            await executeAgent(topic);
        } catch (error) {
            if (error.name === "AbortError") {
                runBadge.textContent = "Stopped";
                runBadge.className =
                    "run-badge stopped";

                showToast(
                    "Workflow cancelled.",
                );
            } else {
                displayError(
                    error.message
                    || "Unexpected error.",
                );
            }
        } finally {
            setRunningState(false);
            abortController = null;
        }
    },
);


cancelButton.addEventListener(
    "click",
    () => {
        if (abortController) {
            abortController.abort();
        }
    },
);


newRunButton.addEventListener(
    "click",
    () => {
        if (abortController) {
            abortController.abort();
        }

        resetInterface();

        topicInput.disabled = false;
        topicInput.value = "";
        topicInput.focus();
    },
);


document
    .querySelectorAll("[data-topic]")
    .forEach(button => {
        button.addEventListener(
            "click",
            () => {
                topicInput.value =
                    button.dataset.topic || "";

                topicInput.focus();
            },
        );
    });


previewTab.addEventListener(
    "click",
    () => {
        previewTab.classList.add("active");
        markdownTab.classList.remove("active");

        articlePreview.hidden = false;
        markdownOutput.hidden = true;
    },
);


markdownTab.addEventListener(
    "click",
    () => {
        markdownTab.classList.add("active");
        previewTab.classList.remove("active");

        markdownOutput.hidden = false;
        articlePreview.hidden = true;
    },
);


copyButton.addEventListener(
    "click",
    async () => {
        if (!finalMarkdown) {
            return;
        }

        await navigator.clipboard.writeText(
            finalMarkdown,
        );

        showToast(
            "Markdown copied.",
        );
    },
);


downloadButton.addEventListener(
    "click",
    event => {
        if (
            downloadButton.classList.contains(
                "disabled",
            )
        ) {
            event.preventDefault();
        }
    },
);

if (downloadZipButton) {
    downloadZipButton.addEventListener(
        "click",
        event => {
            if (
                downloadZipButton.classList.contains(
                    "disabled",
                )
            ) {
                event.preventDefault();
            }
        },
    );
}


async function checkHealth() {
    try {
        const response = await fetch(
            "/api/health",
        );

        if (!response.ok) {
            throw new Error("Server unavailable");
        }

        healthDot.classList.add("online");
        healthText.textContent = "InkFlow-AI Ready";
    } catch {
        healthDot.classList.remove("online");
        healthText.textContent = "Workflow Unavailable";
    }
}


function renderExecutionSummary(summary, metrics) {
    if (!summary) return;

    const elStatus = document.getElementById("sumStatus");
    const elCost = document.getElementById("sumCost");
    const elTokens = document.getElementById("sumTokens");
    const elExecTime = document.getElementById("sumExecTime");
    const elAvgLatency = document.getElementById("sumAvgLatency");
    const elSections = document.getElementById("sumSections");
    const elImages = document.getElementById("sumImages");
    const elSources = document.getElementById("sumSources");
    const elModels = document.getElementById("sumModels");
    const elFallbacks = document.getElementById("sumFallbacks");
    const elViolations = document.getElementById("sumViolations");

    if (elStatus) elStatus.textContent = summary.workflow_status === "completed" ? "🟢 Completed" : "🔴 Failed";
    if (elCost) elCost.textContent = `$${(summary.total_cost ?? 0).toFixed(4)}`;
    if (elTokens) elTokens.textContent = (summary.total_tokens ?? 0).toLocaleString();
    if (elExecTime) elExecTime.textContent = summary.execution_duration || "0.0s";
    if (elAvgLatency) elAvgLatency.textContent = `${summary.average_latency ?? 0}ms`;
    if (elSections) elSections.textContent = summary.sections_generated ?? 0;
    if (elImages) elImages.textContent = summary.images_generated ?? 0;
    if (elSources) elSources.textContent = summary.sources_retrieved ?? 0;
    if (elModels) elModels.textContent = summary.unique_models_used ?? 0;
    if (elFallbacks) elFallbacks.textContent = summary.fallback_count ?? 0;
    if (elViolations) elViolations.textContent = summary.guardrail_violations ?? 0;

    const tbody = document.getElementById("metricsTableBody");
    if (tbody && metrics && metrics.length > 0) {
        const seenSigs = new Set();
        const uniqueMetrics = metrics.filter(m => {
            if (!m) return false;
            const sig = `${m.node_name}_${m.provider}_${m.model}_${m.prompt_tokens}_${m.completion_tokens}_${m.latency_ms}_${m.timestamp}`;
            if (seenSigs.has(sig)) return false;
            seenSigs.add(sig);
            return true;
        });

        tbody.innerHTML = uniqueMetrics.map(m => {
            const isImage = (m.images_generated && m.images_generated > 0) || (m.node_name && m.node_name.includes("image_generator"));
            const isResearch = (m.provider && m.provider.toLowerCase().includes("tavily")) || (m.node_name && m.node_name.includes("research"));

            let promptTok = '—';
            let compTok = '—';
            let totTok = '—';

            if (isImage) {
                promptTok = m.images_generated ? `${m.images_generated} img` : '—';
                compTok = m.resolution || '—';
                totTok = '—';
            } else if (isResearch) {
                promptTok = '1 search';
                compTok = '—';
                totTok = '—';
            } else {
                promptTok = m.prompt_tokens !== null && m.prompt_tokens !== undefined ? m.prompt_tokens.toLocaleString() : '—';
                compTok = m.completion_tokens !== null && m.completion_tokens !== undefined ? m.completion_tokens.toLocaleString() : '—';
                totTok = m.total_tokens !== null && m.total_tokens !== undefined ? m.total_tokens.toLocaleString() : '—';
            }

            const costText = m.estimated_cost !== undefined && m.estimated_cost !== null ? `$${m.estimated_cost.toFixed(4)}` : '—';
            const latencyText = m.latency_ms !== undefined && m.latency_ms !== null ? `${m.latency_ms}ms` : '—';
            const statusText = escapeHtml(m.status || 'completed');
            const fallbackBadge = m.is_fallback ? '<span class="fallback-badge">Fallback</span>' : '';

            return `
                <tr>
                    <td><strong>${escapeHtml(m.node_name || '—')}</strong></td>
                    <td>${escapeHtml(m.provider || '—')}</td>
                    <td>${escapeHtml(m.model || '—')}${fallbackBadge}</td>
                    <td>${promptTok}</td>
                    <td>${compTok}</td>
                    <td>${totTok}</td>
                    <td>${latencyText}</td>
                    <td>${costText}</td>
                    <td><span class="status-badge ${statusText}">${statusText}</span></td>
                </tr>
            `;
        }).join("");
    }
}


async function loadHistoryList() {
    const historyList = document.getElementById("historyList");
    if (!historyList) return;

    try {
        const response = await fetch("/api/history");
        if (!response.ok) return;

        const items = await response.json();
        if (!items || items.length === 0) {
            historyList.innerHTML = '<div class="history-empty">No previous runs found.</div>';
            return;
        }

        historyList.innerHTML = items.map(item => {
            const dateStr = item.created_at ? new Date(item.created_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
            const costStr = item.total_cost !== undefined ? `$${item.total_cost.toFixed(4)}` : '—';
            const tokStr = item.total_tokens ? item.total_tokens.toLocaleString() + ' tok' : '—';
            const durStr = item.execution_duration || '—';

            return `
                <div class="history-card" data-run-id="${escapeHtml(item.run_id)}" onclick="selectHistoryItem('${escapeHtml(item.run_id)}')">
                    <div class="history-title">📄 ${escapeHtml(item.title || 'Untitled Run')}</div>
                    <div class="history-meta">
                        <span>🕒 ${dateStr}</span>
                        <span>⏱ ${durStr}</span>
                        <span>💰 ${costStr}</span>
                    </div>
                </div>
            `;
        }).join("");
    } catch (err) {
        console.error("Failed to load history list:", err);
    }
}


async function selectHistoryItem(runId) {
    if (!runId) return;

    document.querySelectorAll(".history-card").forEach(card => {
        if (card.dataset.runId === runId) {
            card.classList.add("active");
        } else {
            card.classList.remove("active");
        }
    });

    try {
        const response = await fetch(`/api/history/${runId}`);
        if (!response.ok) {
            showToast("Failed to load history item.");
            return;
        }

        const data = await response.json();
        if (data.markdown) {
            displayFinalResult({
                markdown: data.markdown,
                download_url: data.download_url
            });
        }

        if (data.summary) {
            renderExecutionSummary(data.summary, data.metrics || []);
        }

        showToast("Historical blog loaded instantly.");
    } catch (err) {
        console.error("Error fetching history item:", err);
        showToast("Error loading history item.");
    }
}


resetInterface();
checkHealth();
loadHistoryList();