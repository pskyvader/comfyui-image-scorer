const routerLogger = FrontendLogger.create("external_modules.server.frontend.js.index");

const router = {
    routes: {
        compare: {
            template: "/static/comparison/compare.html",
            section: "compare",
        },
        gallery: {
            template: "/static/gallery/gallery.html",
            section: "gallery",
        },
        chains: {
            template: "/static/maps2/maps.html",
            section: "chains",
        },
        tools: {
            template: "/static/database/db.html",
            section: "database",
        },
        db: {
            template: "/static/database/db.html",
            section: "database",
        },
        data: {
            template: "/static/data/transform.html",
            section: "data",
        },
        training: {
            template: "/static/training/training.html",
            section: "training",
        },
        analysis: {
            template: "/static/analysis/analysis.html",
            section: "analysis",
        },
    },

    currentRoute: null,
    _currentInstance: null,
    _currentSection: null,

    async init() {
        window.addEventListener("hashchange", () => this.handleRoute());

        if (!window.location.hash || window.location.hash === "#") {
            window.location.hash = "#compare";
        } else {
            this.handleRoute();
        }

        this.bindNavLinks();
    },

    bindNavLinks() {
        document.querySelectorAll("nav a")
            .forEach((link) => {
                link.addEventListener("click", (e) => {
                    const href = link.getAttribute("href");
                    if (href === "/") {
                        e.preventDefault();
                        window.location.hash = "#compare";
                    }
                });
            });
    },

    async handleRoute() {
        const hash = window.location.hash.substring(1) || "compare";
        const [routeName, queryStr] = hash.split("?");
        const params = new URLSearchParams(queryStr);

        if (this.routes[routeName]) {
            await this.loadRoute(routeName, params);
        } else {
            routerLogger.warn(`Route not found: ${routeName}`);
            window.location.hash = "#compare";
        }
    },

    async loadRoute(routeName, params) {
        const route = this.routes[routeName];
        const contentArea = document.getElementById("app-content");

        contentArea.innerHTML = `
            <div class="flex flex-col items-center justify-center min-h-[400px]">
                <div class="w-12 h-12 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin mb-4"></div>
                <p class="text-gray-500 animate-pulse">Switching to ${routeName}...</p>
            </div>
        `;

        try {
            const response = await fetch(route.template);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const html = await response.text();

            this._destroyCurrentInstance();

            contentArea.innerHTML = html;

            this.updateNavUI(routeName);

            requestAnimationFrame(() => {
                this._createSectionInstance(route.section, params);
            });

            this.currentRoute = routeName;
        } catch (e) {
            routerLogger.error("Router error:", null, e);
            contentArea.innerHTML = `
                <div class="p-12 glass rounded-2xl text-center">
                    <h2 class="text-2xl font-bold text-red-400 mb-2">Navigation Error</h2>
                    <p class="text-gray-400 mb-6">${e.message}</p>
                    <button onclick="window.location.reload()" class="px-6 py-2 bg-purple-600 rounded-lg font-bold">Reload App</button>
                </div>
            `;
        }
    },

    _destroyCurrentInstance() {
        if (this._currentInstance && typeof this._currentInstance.destroy === "function") {
            this._currentInstance.destroy();
        }
        if (this._currentSection === "gallery") {
            window.galleryView = null;
        }
        this._currentInstance = null;
        this._currentSection = null;
    },

    _createSectionInstance(sectionName, params) {
        if (sectionName === "chains") {
            if (window.chainMapUI) {
                window.chainMapUI.init(params);
                this._currentInstance = window.chainMapUI;
                this._currentSection = sectionName;
            }
            return;
        }

        const Cls = window.Sections && window.Sections[sectionName];
        if (Cls) {
            const instance = new Cls();
            if (sectionName === "gallery") {
                window.galleryView = instance;
            }
            if (typeof instance.init === "function") {
                instance.init(params);
            }
            this._currentInstance = instance;
            this._currentSection = sectionName;
        } else {
            routerLogger.error(`Section class not found: ${sectionName}`);
        }
    },

    updateNavUI(routeName) {
        document.querySelectorAll("nav a, .mobile-nav-menu a")
            .forEach((link) => {
                const href = link.getAttribute("href");
                if (!href) {
                    return;
                }
                const isMatch = href === `#${routeName}` || (routeName === "compare" && (href === "/" || href === "#compare" || href === "/#compare"));

                if (isMatch) {
                    link.classList.add("text-white", "bg-white/10", "font-medium");
                    link.classList.remove("text-gray-300");
                } else {
                    link.classList.remove("text-white", "bg-white/10", "font-medium");
                    link.classList.add("text-gray-300");
                }
            });
    },
};

function initMobileNav() {
    const toggle = document.getElementById("nav-toggle-btn");
    const menu = document.getElementById("mobile-nav-menu");
    if (!toggle || !menu) {
        return;
    }
    toggle.addEventListener("click", () => {
        menu.classList.toggle("open");
    });
    menu.querySelectorAll("a")
        .forEach((link) => {
            link.addEventListener("click", () => menu.classList.remove("open"));
        });
}

document.addEventListener("DOMContentLoaded", () => {
    router.init();
    initMobileNav();
});
