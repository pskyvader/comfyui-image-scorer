class ScopedFrontendLogger {
    constructor(sourceName, options = {}) {
        this.sourceName = sourceName;
        this.options = { ...options };
    }

    configure(options = {}) {
        this.options = { ...this.options, ...options };
        return this;
    }

    debug(message, startTimer = null, ...details) {
        FrontendLogger.log(this.sourceName, "debug", message, startTimer, this.options, details);
    }

    info(message, startTimer = null, ...details) {
        FrontendLogger.log(this.sourceName, "info", message, startTimer, this.options, details);
    }

    warn(message, startTimer = null, ...details) {
        FrontendLogger.log(this.sourceName, "warn", message, startTimer, this.options, details);
    }

    error(message, startTimer = null, ...details) {
        FrontendLogger.log(this.sourceName, "error", message, startTimer, this.options, details);
    }

    clear() {
        const target = FrontendLogger.resolveTarget(this.options.target);
        if (target) {
            target.innerHTML = "";
        }
    }
}

class FrontendLogger {
    static allowedExactNames = new Set();
    static allowedPrefixes = [];
    static consoleLevel = "debug";
    static defaultMaxEntries = 100;
    static progressIndicators = ["%", "|", "img/s", "items/s", "[00:", "it/s"];

    static create(sourceName, options = {}) {
        return new ScopedFrontendLogger(sourceName, options);
    }

    static setNameFilters({ exactNames = [], prefixes = [] } = {}) {
        this.allowedExactNames = new Set(exactNames);
        this.allowedPrefixes = [...prefixes];
    }

    static clearNameFilters() {
        this.allowedExactNames = new Set();
        this.allowedPrefixes = [];
    }

    static log(sourceName, level, message, startTimer = null, options = {}, details = []) {
        if (!this.shouldEmit(sourceName)) {
            return;
        }

        const renderedMessage = this.formatMessage(message, startTimer);
        const targetMessage = details.length > 0
            ? `${renderedMessage} ${details.map((detail) => this.serializeDetail(detail)).join(" ")}`
            : renderedMessage;
        this.writeConsole(sourceName, level, renderedMessage, details);
        this.writeTarget(sourceName, level, targetMessage, options);
    }

    static shouldEmit(sourceName) {
        if (this.allowedExactNames.size === 0 && this.allowedPrefixes.length === 0) {
            return true;
        }
        if (this.allowedExactNames.has(sourceName)) {
            return true;
        }
        return this.allowedPrefixes.some((prefix) => sourceName.startsWith(prefix));
    }

    static formatMessage(message, startTimer) {
        let renderedMessage = message;
        if (typeof startTimer === "number") {
            const now = typeof performance !== "undefined" && typeof performance.now === "function"
                ? performance.now()
                : Date.now();
            renderedMessage = `${message} (${((now - startTimer) / 1000).toFixed(4)}s)`;
        }

        return renderedMessage;
    }

    static serializeDetail(detail) {
        if (detail instanceof Error) {
            return detail.stack || detail.message;
        }
        if (typeof detail === "string") {
            return detail;
        }
        try {
            return JSON.stringify(detail);
        } catch {
            return String(detail);
        }
    }

    static writeConsole(sourceName, level, renderedMessage, details) {
        if (this.levelValue(level) < this.levelValue(this.consoleLevel)) {
            return;
        }

        const consoleMethod = this.consoleMethod(level);
        consoleMethod(`[${sourceName}] ${renderedMessage}`, ...details);
    }

    static writeTarget(sourceName, level, renderedMessage, options) {
        const target = this.resolveTarget(options.target);
        if (!target) {
            return;
        }

        const isProgress = options.replaceProgress === true && this.isProgressMessage(renderedMessage);
        const displayMessage = `[${new Date().toLocaleTimeString()}] [${sourceName}] ${renderedMessage}`;
        const lastChild = target.lastElementChild;

        if (isProgress && lastChild && lastChild.dataset?.isProgress === "true") {
            lastChild.textContent = displayMessage;
        } else {
            const line = document.createElement("div");
            line.textContent = displayMessage;
            line.dataset.isProgress = isProgress ? "true" : "false";
            line.dataset.level = level;
            target.appendChild(line);
        }

        const maxEntries = Number.isInteger(options.maxEntries)
            ? options.maxEntries
            : this.defaultMaxEntries;
        while (target.children.length > maxEntries) {
            target.removeChild(target.firstElementChild);
        }

        const shouldAutoScroll = typeof options.shouldAutoScroll === "function"
            ? options.shouldAutoScroll()
            : options.shouldAutoScroll !== false;
        if (shouldAutoScroll) {
            target.scrollTop = target.scrollHeight;
        }

        if (typeof options.afterWrite === "function") {
            options.afterWrite({ sourceName, level, renderedMessage, target });
        }
    }

    static resolveTarget(target) {
        if (typeof target === "function") {
            return target();
        }
        return target || null;
    }

    static isProgressMessage(message) {
        return this.progressIndicators.some((indicator) => message.includes(indicator));
    }

    static levelValue(level) {
        const values = {
            debug: 10,
            info: 20,
            warn: 30,
            error: 40,
        };
        return values[level] ?? values.info;
    }

    static consoleMethod(level) {
        const methods = {
            debug: console.debug ? console.debug.bind(console) : console.log.bind(console),
            info: console.info ? console.info.bind(console) : console.log.bind(console),
            warn: console.warn ? console.warn.bind(console) : console.log.bind(console),
            error: console.error ? console.error.bind(console) : console.log.bind(console),
        };
        return methods[level] ?? console.log.bind(console);
    }
}

window.FrontendLogger = FrontendLogger;
