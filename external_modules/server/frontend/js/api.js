// Base API client - provides shared fetch utilities
class Api {
    constructor(baseUrl = "") {
        this.apiBase = `${baseUrl}/api/v2`;
    }

    async _fetch(path, options = {}) {
        const url = path.startsWith("http") ? path : `${this.apiBase}${path}`;
        const response = await fetch(url, {
            headers: { "Content-Type": "application/json" },
            ...options,
        });
        if (!response.ok) {
            let detail = "";
            try { const e = await response.json(); detail = e.error || e.message || ""; } catch {}
            throw new Error(`${response.status}${detail ? ": " + detail : ""}`);
        }
        return await response.json();
    }

    async _get(path) { return this._fetch(path); }
    async _post(path, body = {}) { return this._fetch(path, { method: "POST", body: JSON.stringify(body) }); }
}

const api = new Api();
window.Api = Api;
