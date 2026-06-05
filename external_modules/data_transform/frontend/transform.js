class DataTransformView {
    init(params) {
        this.container = document.getElementById("transform-container");
        const logArea = this.container?.querySelector("#log-area");
        this.poller = new TaskPoller({
            section: "data",
            logArea,
            logSource: "external_modules.data_transform.frontend.transform",
        });
        this.limitDisplay = this.container?.querySelector("#prepare-limit-display");
        this.batchToggle = this.container?.querySelector("#batch-toggle");
        this.currentLimit = 100;
        if (this.limitDisplay) {
            this.limitDisplay.textContent = `Current: ${this.currentLimit}`;
        }
        this.rebuildLimitDisplay = this.container?.querySelector("#rebuild-limit-display");
        this.currentRebuildLimit = 100;
        this.bindActions();
    }

    destroy() {
        if (this.poller) {
            this.poller.destroy();
        }
    }

    _getBatch() {
        return this.batchToggle ? this.batchToggle.checked : false;
    }

    async _dataPrepare(body) {
        const defaults = {
            limit: 0,
            batch: false,
            rebuild_scores: false,
            text_only: false,
            test_run: false,
        };
        return api._post("/data/prepare", { ...defaults, ...body });
    }

    async _deleteVectors() {
        return api._post("/data/delete-vectors");
    }

    bindActions() {
        const container = this.container;
        if (!container) {
            return;
        }

        container.querySelectorAll("[data-action]")
            .forEach((btn) => {
                btn.addEventListener("click", () => this.handleAction(btn.dataset.action));
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

        container.querySelectorAll("[data-rebuild-limit]")
            .forEach((btn) => {
                btn.addEventListener("click", () => {
                    this.currentRebuildLimit = parseInt(btn.dataset.rebuildLimit, 10);
                    if (this.rebuildLimitDisplay) {
                        this.rebuildLimitDisplay.textContent = `Current: ${this.currentRebuildLimit === 0 ? "All" : this.currentRebuildLimit}`;
                    }
                });
            });

        const clearBtn = container.querySelector("#clear-log-btn");
        if (clearBtn) {
            clearBtn.addEventListener("click", () => this.poller.clearLog());
        }
    }

    async handleAction(action) {
        this.poller.log(`Starting: ${action}...`);
        try {
            let result;
            switch (action) {
                case "prepare-data":
                    result = await this._dataPrepare({ limit: this.currentLimit, batch: this._getBatch() });
                    this.poller.start(result.task_id);
                    return;
                case "rebuild-missing":
                    result = await this._dataPrepare({ rebuild_missing_vectors: true, limit: this.currentRebuildLimit });
                    this.poller.start(result.task_id);
                    return;
                case "delete-vectors":
                    if (!confirm("This will DELETE all vector files from disk. This cannot be undone. Continue?")) {
                        this.poller.log("Delete vectors cancelled by user.");
                        return;
                    }
                    result = await this._deleteVectors();
                    this.poller.start(result.task_id);
                    return;
                case "rebuild-scores":
                    result = await this._dataPrepare({ rebuild_scores: true, limit: this.currentLimit });
                    this.poller.start(result.task_id);
                    return;
                case "prepare-text-only":
                    result = await this._dataPrepare({ text_only: true, limit: this.currentLimit });
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
window.Sections.data = DataTransformView;
