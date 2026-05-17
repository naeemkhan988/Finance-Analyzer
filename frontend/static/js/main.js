/* ═══════════════════════════════════════════════════════════
   Main JavaScript — Finance Analyzer (Claude-Minimal)
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
    if (!toggle || !sidebar) return;

    // Create overlay element for mobile
    let overlay = document.querySelector('.sidebar-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        document.body.appendChild(overlay);
    }

    function openSidebar() {
        sidebar.classList.add('open');
        overlay.classList.add('active');
    }
    function closeSidebar() {
        sidebar.classList.remove('open');
        overlay.classList.remove('active');
    }

    toggle.addEventListener('click', () => {
        if (sidebar.classList.contains('open')) {
            closeSidebar();
        } else {
            openSidebar();
        }
    });

    overlay.addEventListener('click', closeSidebar);

    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && !sidebar.contains(e.target) && !toggle.contains(e.target)) {
            closeSidebar();
        }
    });
}

/* ─── Theme Toggle ──────────────────────────────────────── */
function initTheme() {
    const btn = document.getElementById('themeToggle');
    const saved = localStorage.getItem('theme') || 'light';
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
            indicator.querySelector('.status-text').textContent = 'Model connected';
        })
        .catch(() => {
            indicator.className = 'status-indicator offline';
            indicator.querySelector('.status-text').textContent = 'Model offline';
        });

    setInterval(() => {
        fetch('/api/health')
            .then(res => {
                if (res.ok) {
                    indicator.className = 'status-indicator online';
                    indicator.querySelector('.status-text').textContent = 'Model connected';
                } else { throw new Error('Not OK'); }
            })
            .catch(() => {
                indicator.className = 'status-indicator offline';
                indicator.querySelector('.status-text').textContent = 'Model offline';
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
        toast.style.transform = 'translateX(20px)';
        toast.style.transition = 'all 0.2s ease';
        setTimeout(function() { toast.remove(); }, 200);
    }, 4000);
}

/* ─── New Chat ──────────────────────────────────────────── */
function newChat() {
    fetch('/api/clear-chat', { method: 'POST' })
        .then(function() {
            window.location.href = '/';
        })
        .catch(function() {
            // Even if clear fails, navigate to chat
            window.location.href = '/';
        });
}
