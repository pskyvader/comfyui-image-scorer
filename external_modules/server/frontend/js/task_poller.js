class TaskPoller {
    constructor({ section, logArea, logSource, replaceProgress, maxEntries, onDone, onError, onCancelled, autoScroll }) {
        this.section = section;
        this.logArea = logArea;
        this.logger = FrontendLogger.create(logSource || `external_modules.${section}.frontend`, {
            target: () => this.logArea,
            replaceProgress: replaceProgress !== false,
            maxEntries: maxEntries || 100,
            shouldAutoScroll: () => this._isAutoScroll,
        });
        this.onDone = onDone;
        this.onError = onError;
        this.onCancelled = onCancelled;
        this.taskId = null;
        this.pollTimer = null;
        this.lastLogLen = 0;
        this._isAutoScroll = autoScroll !== false;
        this._sseSource = null;
        this._sseBuffer = [];
        this._sseTaskId = null;
        this._connectSSE();
        this._bindScrollListener();
    }

    _connectSSE() {
        if (this._sseSource) return;
        const base = window.location.origin;
        this._sseSource = new EventSource(`${base}/api/v2/logs/stream`);
        this._sseSource.onmessage = (event) => {
            const line = event.data;
            if (this._sseTaskId) {
                if (line.includes(this._sseTaskId) || this._activeTaskLineMatch(line)) {
                    this.logger.info(line);
                }
            }
        };
        this._sseSource.onerror = () => {
            this._sseSource.close();
            this._sseSource = null;
            setTimeout(() => this._connectSSE(), 3000);
        };
    }

    _activeTaskLineMatch(line) {
        return this.taskId && line.includes(this.taskId);
    }

    _bindScrollListener() {
        if (!this.logArea) return;
        this.logArea.addEventListener("scroll", () => {
            const el = this.logArea;
            const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 50;
            this._isAutoScroll = atBottom;
        });
    }

    log(msg) {
        this.logger.info(msg);
    }

    clearLog() {
        this.logger.clear();
    }

    start(taskId) {
        this.taskId = taskId;
        this._sseTaskId = taskId;
        this.lastLogLen = 0;
        this._isAutoScroll = true;
        this.log(`Task started: ${taskId}`);
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
        }
        this.pollTimer = setInterval(() => this._poll(), 2000);
    }

    stop() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
            this.pollTimer = null;
        }
        this._sseTaskId = null;
        this.taskId = null;
    }

    async _poll() {
        if (!this.taskId) return;
        try {
            const res = await api._get(`/${this.section}/task/${this.taskId}?since=${this.lastLogLen}`);
            if (res._log_new && res._log_new.length > 0) {
                res._log_new.forEach(line => this.log(line));
                this.lastLogLen = res._log_total || (this.lastLogLen + res._log_new.length);
            }
            if (res.status === "done") {
                this.stop();
                this.log(`Done: ${JSON.stringify(res.result || {})}`);
                if (this.onDone) this.onDone(res);
            } else if (res.status === "error") {
                this.stop();
                this.log(`Error: ${res.error || "unknown"}`);
                if (this.onError) this.onError(res);
            } else if (res.status === "cancelled") {
                this.stop();
                this.log("Task cancelled");
                if (this.onCancelled) this.onCancelled(res);
            }
        } catch (e) {
            this.log(`Poll error: ${e.message}`);
        }
    }

    scrollIntoView() {
        if (this.logArea) {
            this.logArea.scrollIntoView({ behavior: "smooth", block: "start" });
            setTimeout(() => {
                this.logArea.scrollTop = this.logArea.scrollHeight;
            }, 100);
        }
    }

    destroy() {
        this.stop();
        if (this._sseSource) {
            this._sseSource.close();
            this._sseSource = null;
        }
    }
}
