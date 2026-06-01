class AnalysisView {
    init(params) {
        this.container = document.getElementById("analysis-container");
        const logArea = this.container?.querySelector("#log-area");
        this.poller = new TaskPoller({
            section: "analysis",
            logArea,
            logSource: "external_modules.analysis.frontend.analysis",
            onDone: (res) => this._onTaskDone(res),
            onError: (res) => this._onTaskError(res),
            onCancelled: (res) => this._onTaskCancelled(res),
        });
        this.resultPanel = this.container?.querySelector("#result-panel");
        this.resultContent = this.container?.querySelector("#result-content");
        this.bindActions();
        this.refreshStats();
    }

    destroy() {
        if (this.poller) this.poller.destroy();
    }

    async _getStats() {
        return api._get("/analysis/stats");
    }

    async _analyzeParameters() {
        return api._post("/analysis/analyze-parameters");
    }

    async _analyzeMatrix() {
        return api._post("/analysis/analyze-matrix");
    }

    _onTaskDone(res) {
        const resultHtml = `<pre class="text-xs text-green-400">${JSON.stringify(res.result || {}, null, 2)}</pre>`;
        this.showResult(resultHtml);
    }

    _onTaskError(res) {
        this.showResult(`<div class="text-red-400">Error: ${res.error || "unknown"}</div>`);
    }

    _onTaskCancelled(res) {
        this.showResult(`<div class="text-yellow-400">Task cancelled</div>`);
    }

    bindActions() {
        const container = this.container;
        if (!container) return;

        container.querySelectorAll("[data-action]").forEach(btn => {
            btn.addEventListener("click", () => this.handleAction(btn.dataset.action));
        });

        const refreshBtn = container.querySelector("#refresh-stats-btn");
        if (refreshBtn) refreshBtn.addEventListener("click", () => this.refreshStats());

        const clearLogBtn = container.querySelector("#clear-log-btn");
        if (clearLogBtn) clearLogBtn.addEventListener("click", () => this.poller.clearLog());

        const clearResultBtn = container.querySelector("#clear-result-btn");
        if (clearResultBtn) clearResultBtn.addEventListener("click", () => this.clearResult());
    }

    showResult(html) {
        if (this.resultPanel) this.resultPanel.style.display = "block";
        if (this.resultContent) this.resultContent.innerHTML = html;
    }

    clearResult() {
        if (this.resultPanel) this.resultPanel.style.display = "none";
        if (this.resultContent) this.resultContent.innerHTML = "";
    }

    async refreshStats() {
        this.poller.log("Loading stats...");
        try {
            const data = await this._getStats();
            this.renderStats(data);
            this.renderHistograms(data);
            this.poller.log(`Stats loaded: ${data.total_images} images, ${data.total_comparisons} comparisons`);
        } catch (e) {
            this.poller.log(`Stats error: ${e.message}`);
        }
    }

    renderStats(data) {
        const set = (id, val) => {
            const el = this.container?.querySelector(id);
            if (el) el.textContent = val != null ? String(val) : "—";
        };
        set("#stat-images", data.total_images);
        set("#stat-comparisons", data.total_comparisons);
        set("#stat-avg-sigma", data.avg_sigma);
        set("#stat-chains", data.total_chains);

        if (data.sigma_summary) {
            const ss = this.container?.querySelector("#sigma-summary");
            if (ss) ss.textContent = data.sigma_summary;
        }

        const renderList = (id, items, labelKey, valKey) => {
            const ul = this.container?.querySelector(id);
            if (!ul) return;
            ul.innerHTML = "";
            (items || []).forEach(item => {
                const li = document.createElement("li");
                li.textContent = `${item[labelKey]}: ${item[valKey]}`;
                ul.appendChild(li);
            });
        };

        renderList("#top-list", data.top_images, "filename", "score");
        renderList("#bottom-list", data.bottom_images, "filename", "score");
        renderList("#least-compared-list", data.least_compared, "filename", "comparison_count");
    }

    renderHistograms(data) {
        const containers = {
            mu: { id: "mu-chart", data: data.mu_buckets, color: "#8b5cf6" },
            sigma: { id: "sigma-chart", data: data.sigma_buckets, color: "#10b981" },
            score: { id: "score-chart", data: data.score_buckets, color: "#f59e0b" },
            comp: { id: "comp-chart", data: data.comp_buckets, color: "#3b82f6" },
            chain: { id: "chain-chart", data: data.chain_buckets, color: "#ec4899" },
        };

        Object.values(containers).forEach(c => {
            const el = this.container?.querySelector(`#${c.id}`);
            if (!el) return;
            if (!c.data || Object.keys(c.data).length === 0) {
                el.innerHTML = '<div class="text-gray-500 text-xs p-2">No data</div>';
                return;
            }
            const entries = Object.entries(c.data);
            const maxVal = Math.max(...entries.map(([, v]) => v), 1);
            const bars = entries.map(([label, val]) => {
                const pct = (val / maxVal) * 100;
                return `<div style="display:flex;align-items:center;gap:4px;margin-bottom:2px;">
                    <span style="width:60px;flex-shrink:0;font-size:10px;color:#94a3b8;text-align:right;">${label}</span>
                    <div style="flex:1;height:14px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;">
                        <div style="width:${pct}%;height:100%;background:${c.color};border-radius:3px;opacity:0.8;"></div>
                    </div>
                    <span style="width:30px;font-size:10px;color:#e2e8f0;text-align:right;">${val}</span>
                </div>`;
            }).join("");
            el.innerHTML = `<div style="font-size:11px;font-weight:600;color:#64748b;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;">${c.id}</div>${bars}`;
        });
    }

    async handleAction(action) {
        this.poller.log(`Starting: ${action}...`);
        try {
            let result;
            switch (action) {
                case "analyze-parameters":
                    result = await this._analyzeParameters();
                    this.poller.start(result.task_id);
                    return;
                case "analyze-matrix":
                    result = await this._analyzeMatrix();
                    this.poller.start(result.task_id);
                    return;
                default:
                    this.poller.log(`Unknown action: ${action}`);
            }
        } catch (e) {
            this.poller.log(`Error: ${e.message}`);
        }
    }
}

window.Sections = window.Sections || {};
window.Sections.analysis = AnalysisView;
