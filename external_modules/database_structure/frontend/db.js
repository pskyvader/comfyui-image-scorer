class DbView {
    init(params) {
        this.container = document.getElementById("db-container");
        this.logArea = this.container?.querySelector("#log-area");
        this.logger = FrontendLogger.create("external_modules.database_structure.frontend.db", {
            target: () => this.logArea,
            replaceProgress: true,
            maxEntries: 100,
            shouldAutoScroll: () => this.isAutoScrollEnabled,
        });
        this.taskId = null;
        this.pollTimer = null;
        this.lastLogLen = 0;
        this.isAutoScrollEnabled = true; // Track if autoscroll should be active
        this.bindActions();
        this.bindScrollListener();
    }

    destroy() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer); this.pollTimer = null;
        }
    }

    // ── API methods ──────────────────────────────────────────────────

    async _dbAction(action, body) {
        const actions = {
            "sync-all": "sync-all",
            "rebuild-db": "rebuild-db",
            "reset-ratings": "reset-ratings",
            "normalize-comparisons": "normalize-comparisons",
            "cleanup-orphans": "cleanup-orphans",
            "deduplicate": "deduplicate",
            "remove-vectors": "remove-vectors",
        };
        const path = actions[action];
        if (!path) {
            throw new Error(`Unknown action: ${action}`);
        }
        const query = "";
        // if (body) {
        //     query = "?" + new URLSearchParams(body)
        //         .toString();
        // }
        return api._post(`/database/${path}${query}`, body);
    }

    async _dbRemoveVectors() {
        return api._post("/database/remove-vectors");
    }

    async _getDbTask(taskId, offset) {
        return api._get(`/database/task/${taskId}?offset=${offset}`);
    }

    bindActions() {
        const container = this.container;
        if (!container) {
            return;
        }
        container.querySelectorAll("[data-action]")
            .forEach((btn) => {
                btn.addEventListener("click", () => {
                    const action = btn.dataset.action;
                    this.handleAction(action);
                });
            });
        const clearBtn = container.querySelector("#clear-log-btn");
        if (clearBtn) {
            clearBtn.addEventListener("click", () => this.clearLog());
        }
    }

    bindScrollListener() {
        if (!this.logArea) {
            return;
        }
        this.logArea.addEventListener("scroll", () => {
            // Check if user scrolled to bottom (within 50px threshold for mobile)
            const isAtBottom = this.logArea.scrollTop + this.logArea.clientHeight >= this.logArea.scrollHeight - 50;
            this.isAutoScrollEnabled = isAtBottom;
        });
    }

    log(msg) {
        this.logger.info(msg);
    }

    clearLog() {
        this.logger.clear();
    }

    scrollToLog() {
        if (this.logArea) {
            this.logArea.scrollIntoView({ behavior: "smooth", block: "start" });
            // Also scroll to bottom of the log area
            setTimeout(() => {
                this.logArea.scrollTop = this.logArea.scrollHeight;
            }, 100);
        }
    }

    startPolling(taskId) {
        this.taskId = taskId;
        this.lastLogLen = 0;
        this.isAutoScrollEnabled = true; // Enable autoscroll when task starts
        this.log(`Task started: ${taskId}`);
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
        }
        this.pollTimer = setInterval(() => this.poll(), 1000);
    }

    async poll() {
        if (!this.taskId) {
            return;
        }
        try {
            const res = await this._getDbTask(this.taskId, this.lastLogLen);
            if (res._log_new && res._log_new.length > 0) {
                res._log_new.forEach(line => this.log(line));
                this.lastLogLen = res._log_total || (this.lastLogLen + res._log_new.length);
            }
            if (res.status === "done") {
                clearInterval(this.pollTimer);
                this.pollTimer = null;
                this.log(`Done: ${JSON.stringify(res.result || {})}`);
                this.taskId = null;
            } else if (res.status === "error") {
                clearInterval(this.pollTimer);
                this.pollTimer = null;
                this.log(`Error: ${res.error || "unknown"}`);
                this.taskId = null;
            }
        } catch (e) {
            this.log(`Poll error: ${e.message}`);
        }
    }

    async handleAction(action) {
        this.log(`Starting: ${action}...`);
        this.scrollToLog(); // Navigate to log area when action starts
        try {
            let result;
            switch (action) {
                case "sync-all":
                    result = await this._dbAction("sync-all");
                    this.startPolling(result.task_id);
                    return;
                case "rebuild-db":
                    result = await this._dbAction("rebuild-db");
                    this.startPolling(result.task_id);
                    return;
                case "reset-ratings":
                    result = await this._dbAction("reset-ratings");
                    this.log(`Reset: ${result.status}`);
                    break;
                case "normalize-comparisons":
                    result = await this._dbAction("normalize-comparisons");
                    this.log(`Normalized: ${JSON.stringify(result.stats || result)}`);
                    break;
                case "cleanup-orphans-dry":
                    result = await this._dbAction("cleanup-orphans", { dry_run: true });
                    this.log(`Orphans (dry): ${JSON.stringify(result.result || result)}`);
                    break;
                case "cleanup-orphans":
                    result = await this._dbAction("cleanup-orphans", { dry_run: false });
                    this.log(`Cleaned up: ${JSON.stringify(result.result || result)}`);
                    break;
                case "deduplicate-dry":
                    result = await this._dbAction("deduplicate", { dry_run: true, limit: 0 });
                    this.log(`Dedup (dry): ${JSON.stringify(result.result || result)}`);
                    break;
                case "deduplicate":
                    result = await this._dbAction("deduplicate", { dry_run: false, limit: 1000 });
                    this.log(`Deduplicated: ${JSON.stringify(result.result || result)}`);
                    break;
                default:
                    this.log(`Unknown action: ${action}`);
            }
        } catch (e) {
            this.log(`Error: ${e.message}`);
        }
    }
}

window.Sections = window.Sections || {};
window.Sections.database = DbView;
