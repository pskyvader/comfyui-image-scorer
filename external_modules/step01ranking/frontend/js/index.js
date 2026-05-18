/**
 * SPA Router for Image Ranking System
 */

const router = {
    routes: {
        'compare': {
            template: '/templates/compare.html',
            controller: (params) => {
                if (typeof window.compareMode !== 'undefined') {
                    window.compareMode.init(params);
                } else {
                    console.error("compareMode not found");
                }
            }
        },
        'gallery': {
            template: '/templates/gallery.html',
            controller: (params) => {
                if (typeof window.galleryView !== 'undefined') {
                    window.galleryView.init(params);
                } else {
                    console.error("galleryView not found");
                }
            }
        },
        'chains': {
            template: '/templates/chains.html',
            controller: (params) => {
                if (typeof window.chainMapUI !== 'undefined') {
                    window.chainMapUI.init(params);
                } else {
                    console.error("chainMapUI not found");
                }
            }
        }
    },

    currentRoute: null,

    async init() {
        window.addEventListener('hashchange', () => this.handleRoute());
        
        // Initial route
        if (!window.location.hash || window.location.hash === '#') {
            window.location.hash = '#compare';
        } else {
            this.handleRoute();
        }
        
        // Global navigation listener
        this.bindNavLinks();
    },

    bindNavLinks() {
        document.querySelectorAll('nav a').forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                if (href === '/') {
                    e.preventDefault();
                    window.location.hash = '#compare';
                }
            });
        });
    },

    async handleRoute() {
        const hash = window.location.hash.substring(1) || 'compare';
        const [routeName, queryStr] = hash.split('?');
        const params = new URLSearchParams(queryStr);
        
        if (this.routes[routeName]) {
            await this.loadRoute(routeName, params);
        } else {
            console.warn(`Route not found: ${routeName}`);
            window.location.hash = '#compare';
        }
    },

    async loadRoute(routeName, params) {
        const route = this.routes[routeName];
        const contentArea = document.getElementById('app-content');
        
        // Show loading spinner
        contentArea.innerHTML = `
            <div class="flex flex-col items-center justify-center min-h-[400px]">
                <div class="w-12 h-12 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin mb-4"></div>
                <p class="text-gray-500 animate-pulse">Switching to ${routeName}...</p>
            </div>
        `;
        
        try {
            const response = await fetch(route.template);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const html = await response.text();
            
            // Clean up old controllers if they have destroy methods
            if (this.currentRoute && this.routes[this.currentRoute].cleanup) {
                this.routes[this.currentRoute].cleanup();
            }

            contentArea.innerHTML = html;
            
            // Update UI
            this.updateNavUI(routeName);

            // Give the DOM a moment to settle, then init controller
            requestAnimationFrame(() => {
                if (route.controller) {
                    route.controller(params);
                }
            });
            
            this.currentRoute = routeName;
        } catch (e) {
            console.error(`Router error:`, e);
            contentArea.innerHTML = `
                <div class="p-12 glass rounded-2xl text-center">
                    <h2 class="text-2xl font-bold text-red-400 mb-2">Navigation Error</h2>
                    <p class="text-gray-400 mb-6">${e.message}</p>
                    <button onclick="window.location.reload()" class="px-6 py-2 bg-purple-600 rounded-lg font-bold">Reload App</button>
                </div>
            `;
        }
    },

    updateNavUI(routeName) {
        document.querySelectorAll('nav a').forEach(link => {
            const href = link.getAttribute('href');
            const isMatch = href && (href.includes(routeName) || (routeName === 'compare' && (href === '/' || href === '#compare' || href === '/#compare')));
            
            if (isMatch) {
                link.classList.add('text-white', 'bg-white/10', 'font-medium');
                link.classList.remove('text-gray-300');
            } else {
                link.classList.remove('text-white', 'bg-white/10', 'font-medium');
                link.classList.add('text-gray-300');
            }
        });
    }
};

document.addEventListener('DOMContentLoaded', () => router.init());
