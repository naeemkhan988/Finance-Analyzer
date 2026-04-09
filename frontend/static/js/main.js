/* ═══════════════════════════════════════════════════════════
   Main JavaScript — Finance Analyzer
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initTheme();
    checkApiHealth();
});

/* ─── Sidebar Toggle ────────────────────────────────────── */
function initSidebar() {
    const toggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    if (toggle && sidebar) {
        toggle.addEventListener('click', () => { sidebar.classList.toggle('open'); });
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 && !sidebar.contains(e.target) && !toggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }
}

/* ─── Theme Toggle ──────────────────────────────────────── */
function initTheme() {
    const btn = document.getElementById('themeToggle');
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
    if (btn) {
        btn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            updateThemeIcon(next);
        });
    }
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    const icon = btn.querySelector('i');
    icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

/* ─── API Health Check ──────────────────────────────────── */
function checkApiHealth() {
    const indicator = document.getElementById('api-status');
    if (!indicator) return;

    fetch('/api/health')
        .then(res => res.json())
        .then(() => {
            indicator.className = 'status-indicator online';
            indicator.querySelector('.status-text').textContent = 'API Online';
        })
        .catch(() => {
            indicator.className = 'status-indicator offline';
            indicator.querySelector('.status-text').textContent = 'API Offline';
        });

    setInterval(() => {
        fetch('/api/health')
            .then(res => {
                if (res.ok) {
                    indicator.className = 'status-indicator online';
                    indicator.querySelector('.status-text').textContent = 'API Online';
                } else { throw new Error('Not OK'); }
            })
            .catch(() => {
                indicator.className = 'status-indicator offline';
                indicator.querySelector('.status-text').textContent = 'API Offline';
            });
    }, 30000);
}

/* ─── Toast Notifications ───────────────────────────────── */
function showToast(message, type) {
    type = type || 'info';
    var container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    var icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle',
        warning: 'fa-exclamation-triangle'
    };

    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.innerHTML = '<i class="fas ' + (icons[type] || icons.info) + '"></i><span>' + message + '</span>';
    container.appendChild(toast);

    setTimeout(function() {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(function() { toast.remove(); }, 300);
    }, 4000);
}
