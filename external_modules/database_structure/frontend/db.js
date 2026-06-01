class DbView {
    init(params) {
        this.container = document.getElementById("db-container");
        const logArea = this.container?.querySelector("#log-area");
        this.poller = new TaskPoller({
            section: "database",
            logArea,
            logSource: "external_modules.database_structure.frontend.db",
            replaceProgress: true,
            autoScroll: true,
        });
        this.bindActions();
    }

    destroy() {
        if (this.poller) this.poller.destroy();
    }

    async _dbAction(action, body) {
        const actions = {
            "sync-all": "sync-all",
            "rebuild-db": "rebuild-db",
            "reset-ratings": "reset-ratings",
            "normalize-comparisons": "normalize-comparisons",
            "cleanup-orphans": "cleanup-orphans",
            "deduplicate": "deduplicate",
        };
        const path = actions[action];
        if (!path) throw new Error(`Unknown action: ${action}`);
        return api._post(`/database/${path}`, body);
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
        this.poller.scrollIntoView();
        try {
            let result;
            switch (action) {
                case "sync-all":
                    result = await this._dbAction("sync-all");
                    this.poller.start(result.task_id);
                    return;
                case "rebuild-db":
                    result = await this._dbAction("rebuild-db");
                    this.poller.start(result.task_id);
                    return;
                case "reset-ratings":
                    result = await this._dbAction("reset-ratings");
                    this.poller.log(`Reset: ${result.status}`);
                    break;
                case "normalize-comparisons":
                    result = await this._dbAction("normalize-comparisons");
                    this.poller.log(`Normalized: ${JSON.stringify(result.stats || result)}`);
                    break;
                case "cleanup-orphans-dry":
                    result = await this._dbAction("cleanup-orphans", { dry_run: true });
                    this.poller.log(`Orphans (dry): ${JSON.stringify(result.result || result)}`);
                    break;
                case "cleanup-orphans":
                    result = await this._dbAction("cleanup-orphans", { dry_run: false });
                    this.poller.log(`Cleaned up: ${JSON.stringify(result.result || result)}`);
                    break;
                case "deduplicate-dry":
                    result = await this._dbAction("deduplicate", { dry_run: true, limit: 0 });
                    this.poller.log(`Dedup (dry): ${JSON.stringify(result.result || result)}`);
                    break;
                case "deduplicate":
                    result = await this._dbAction("deduplicate", { dry_run: false, limit: 1000 });
                    this.poller.log(`Deduplicated: ${JSON.stringify(result.result || result)}`);
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
window.Sections.database = DbView;
