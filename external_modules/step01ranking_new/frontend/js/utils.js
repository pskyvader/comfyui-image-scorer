// Utility functions for ranking system

class Utils {
    // Format a score to 3 decimal places
    static formatScore(score) {
        return (Math.round(score * 1000) / 1000).toFixed(3);
    }

    // Get tier name from score
    static getTierName(score) {
        const tier = Math.floor(score * 10);
        const min = (tier / 10).toFixed(1);
        const max = ((tier + 1) / 10).toFixed(1);
        return `Tier ${tier} (${min}-${max})`;
    }

    // Debounce function
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Show toast notification
    static showToast(message, type = "info", duration = 3000) {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.add("show");
        }, 10);

        setTimeout(() => {
            toast.classList.remove("show");
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // Add CSS for toast
    static initializeToastStyles() {
        if (!document.getElementById("toast-styles")) {
            const style = document.createElement("style");
            style.id = "toast-styles";
            style.textContent = `
                .toast {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: #1e1e2d;
                    color: white;
                    padding: 1rem 1.5rem;
                    border-radius: 8px;
                    border: 1px solid rgba(255,255,255,0.1);
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
                    opacity: 0;
                    transition: opacity 0.3s ease, transform 0.3s ease;
                    transform: translateY(20px);
                    z-index: 9999;
                }
                .toast.show {
                    opacity: 1;
                    transform: translateY(0);
                }
                .toast-success { border-left: 4px solid #10b981; }
                .toast-error { border-left: 4px solid #ef4444; }
                .toast-info { border-left: 4px solid #3b82f6; }
            `;
            document.head.appendChild(style);
        }
    }

    // Preload image for performance
    static preloadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    }
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    Utils.initializeToastStyles();
});
