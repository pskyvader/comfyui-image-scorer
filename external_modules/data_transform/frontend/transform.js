class DataTransformView {
    init(params) {
        this.container = document.getElementById("transform-container");
        this.logArea = this.container?.querySelector("#log-area");
        this.logger = FrontendLogger.create("external_modules.data_transform.frontend.transform", {
            target: () => this.logArea,
            maxEntries: 100,
        });
        this.limitDisplay = this.container?.querySelector("#prepare-limit-display");
        this.currentLimit = 0;
        this.taskId = null;
        this.pollTimer = null;
        this.lastLogLen = 0;
        this.bindActions();
    }

    destroy() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer); this.pollTimer = null;
        }
    }

    // ── API methods ──────────────────────────────────────────────────

    async _dataPrepare(body) {
        const defaults = {
            rebuild: false,
            limit: 0,
            rebuild_scores: false,
            rebuild_from_splits: false,
            text_only: false,
            test_run: false,

        };
        return api._post("/data/prepare", { ...defaults, ...body });
    }

    async _dataScanImport() {
        return api._post("/data/scan-import");
    }

    async _getDataTask(taskId, offset) {
        return api._get(`/data/task/${taskId}?offset=${offset}`);
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

        container.querySelectorAll("[data-prepare-limit]")
            .forEach((btn) => {
                btn.addEventListener("click", () => {
                    this.currentLimit = parseInt(btn.dataset.prepareLimit, 10);
                    if (this.limitDisplay) {
                        this.limitDisplay.textContent = `Current: ${this.currentLimit === 0 ? "All" : this.currentLimit}`;
                    }
                });
            });

        const clearBtn = container.querySelector("#clear-log-btn");
        if (clearBtn) {
            clearBtn.addEventListener("click", () => this.clearLog());
        }
    }

    log(msg) {
        this.logger.info(msg);
    }

    clearLog() {
        this.logger.clear();
    }

    startPolling(taskId) {
        this.taskId = taskId;
        this.lastLogLen = 0;
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
            const res = await this._getDataTask(this.taskId, this.lastLogLen);
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
        try {
            let result;
            switch (action) {
                case "prepare-data":
                    result = await this._dataPrepare({ limit: this.currentLimit });
                    this.startPolling(result.task_id);
                    return;
                case "prepare-data-rebuild":
                    result = await this._dataPrepare({ rebuild: true, limit: this.currentLimit });
                    this.startPolling(result.task_id);
                    return;
                case "prepare-scores":
                    result = await this._dataPrepare({ rebuild_scores: true, limit: this.currentLimit });
                    this.startPolling(result.task_id);
                    return;
                case "prepare-from-splits":
                    result = await this._dataPrepare({ rebuild_from_splits: true });
                    this.startPolling(result.task_id);
                    return;
                case "prepare-text-only":
                    result = await this._dataPrepare({ text_only: true, limit: this.currentLimit });
                    this.startPolling(result.task_id);
                    return;
                case "scan-import":
                    result = await this._dataScanImport();
                    this.log(`Scan: ${JSON.stringify(result.stats || result)}`);
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
window.Sections.data = DataTransformView;
