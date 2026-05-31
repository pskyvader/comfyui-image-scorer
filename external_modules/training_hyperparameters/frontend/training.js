class TrainingView {
    init(params) {
        this.container = document.getElementById("training-container");
        this.logArea = this.container?.querySelector("#log-area");
        this.logger = FrontendLogger.create("external_modules.training_hyperparameters.frontend.training", {
            target: () => this.logArea,
            maxEntries: 100,
        });
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

    async _trainingTrain(body) {
        return api._post("/training/train", body);
    }

    async _trainingHpo(body = {}) {
        return api._post("/training/hpo", body);
    }

    async _trainingRemoveModels() {
        return api._post("/training/remove-models");
    }

    async _trainingResetConfig() {
        return api._post("/training/reset");
    }

    async _getTrainingConfig() {
        return api._get("/training/config");
    }

    async _getTrainingTask(taskId, offset) {
        return api._get(`/training/task/${taskId}?offset=${offset}`);
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

    log(msg) {
        this.logger.info(msg);
    }

    clearLog() {
        this.logger.clear();
    }

    startPolling(taskId, section = "training") {
        this.taskId = taskId;
        this.lastLogLen = 0;
        this.log(`Task started: ${taskId}`);
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
        }
        this.pollTimer = setInterval(() => this.poll(section), 1000);
    }

    async poll(section) {
        if (!this.taskId) {
            return;
        }
        try {
            const getTask = section === "training" ? this._getTrainingTask : this._getDataTask;
            const res = await getTask(this.taskId, this.lastLogLen);
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
                case "train-top":
                    result = await this._trainingTrain({ strategy: "top1" });
                    this.startPolling(result.task_id);
                    return;
                case "optimize-hpo":
                    result = await this._trainingTrain({ strategy: "optimize" });
                    this.startPolling(result.task_id);
                    return;
                case "hpo-cycle":
                    result = await this._trainingHpo();
                    this.startPolling(result.task_id);
                    return;
                case "get-training-config":
                    result = await this._getTrainingConfig();
                    this.log(`Config: ${JSON.stringify(result.config || result, null, 2)}`);
                    break;
                case "remove-models":
                    result = await this._trainingRemoveModels();
                    this.log(`Models removed: ${result.status}`);
                    break;
                case "reset-config":
                    result = await this._trainingResetConfig();
                    this.log(`Config reset: ${result.status}`);
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
window.Sections.training = TrainingView;
