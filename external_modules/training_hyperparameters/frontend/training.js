class TrainingView {
    init(params) {
        this.container = document.getElementById("training-container");
        const logArea = this.container?.querySelector("#log-area");
        this.poller = new TaskPoller({
            section: "training",
            logArea,
            logSource: "external_modules.training_hyperparameters.frontend.training",
        });
        this.bindActions();
    }

    destroy() {
        if (this.poller) this.poller.destroy();
    }

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

    bindActions() {
        const container = this.container;
        if (!container) return;
        container.querySelectorAll("[data-action]").forEach((btn) => {
            btn.addEventListener("click", () => this.handleAction(btn.dataset.action));
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
                case "train-top":
                    result = await this._trainingTrain({ strategy: "top1" });
                    this.poller.start(result.task_id);
                    return;
                case "optimize-hpo":
                    result = await this._trainingTrain({ strategy: "optimize" });
                    this.poller.start(result.task_id);
                    return;
                case "hpo-cycle":
                    result = await this._trainingHpo();
                    this.poller.start(result.task_id);
                    return;
                case "get-training-config":
                    result = await this._getTrainingConfig();
                    this.poller.log(`Config: ${JSON.stringify(result.config || result, null, 2)}`);
                    break;
                case "remove-models":
                    result = await this._trainingRemoveModels();
                    this.poller.log(`Models removed: ${result.status}`);
                    break;
                case "reset-config":
                    result = await this._trainingResetConfig();
                    this.poller.log(`Config reset: ${result.status}`);
                    break;
                default:
                    this.poller.log(`Unknown action: ${action}`);
            }
        } catch (e) {
            this.poller.log(`Error: ${e.message}`);
        }
    }
}

window.Sections = window.Sections || {};
window.Sections.training = TrainingView;
