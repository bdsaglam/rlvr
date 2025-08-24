// JavaScript functionality for the Multi-Hop QA Results Inspector

// Tab management
function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.add('hidden');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content
    const targetContent = document.getElementById(tabName + '-content');
    if (targetContent) {
        targetContent.classList.remove('hidden');
    }
    
    // Add active class to selected tab button
    const targetButton = document.getElementById('tab-' + tabName);
    if (targetButton) {
        targetButton.classList.add('active');
    }
    
    // Initialize charts if showing overview
    if (tabName === 'overview') {
        initializeCharts();
    }
}

// Filter management
let currentFilters = {
    status: '',
    hops: '',
    minReward: ''
};

function filterResults(status) {
    // Update filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    currentFilters.status = status;
    applyFilters();
}

function applyFilters() {
    // Get filter values
    const hopFilter = document.querySelector('select[name="hop_filter"]');
    const minRewardFilter = document.querySelector('input[name="min_reward"]');
    
    if (hopFilter) currentFilters.hops = hopFilter.value;
    if (minRewardFilter) currentFilters.minReward = minRewardFilter.value;
    
    // Build query parameters
    const params = new URLSearchParams();
    if (currentFilters.status === 'correct') params.set('correct_only', 'true');
    if (currentFilters.status === 'incorrect') params.set('incorrect_only', 'true');
    if (currentFilters.hops) params.set('min_hops', currentFilters.hops);
    if (currentFilters.minReward) params.set('min_reward', currentFilters.minReward);
    
    // Make HTMX request to filter results
    htmx.ajax('GET', '/filter?' + params.toString(), '#table-content');
}

// Chart initialization
function initializeCharts() {
    // Initialize reward distribution chart if it exists
    const chartCanvas = document.getElementById('rewardChart');
    if (chartCanvas && typeof Chart !== 'undefined') {
        // This would be populated with actual data from the server
        initializeRewardChart(chartCanvas);
    }
}

function initializeRewardChart(canvas) {
    const ctx = canvas.getContext('2d');
    
    // Sample data - in a real app this would come from the server
    const rewardChart = new Chart(ctx, {
        type: 'histogram',
        data: {
            datasets: [{
                label: 'Reward Distribution',
                data: [], // This would be populated with actual reward values
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Reward Value'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                }
            }
        }
    });
}

// Search functionality
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', function(e) {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                performSearch(e.target.value);
            }, 300); // Debounce search
        });
    }
}

function performSearch(query) {
    if (query.length < 2) return;
    
    // Make HTMX request to search
    htmx.ajax('GET', '/search?q=' + encodeURIComponent(query), '#search-results');
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Number keys 1-3 for tab navigation
    if (e.altKey) {
        switch(e.key) {
            case '1':
                showTab('overview');
                break;
            case '2':
                showTab('table');
                break;
            case '3':
                showTab('errors');
                break;
        }
    }
});

// Tooltip functionality
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function(e) {
            showTooltip(e.target, e.target.getAttribute('data-tooltip'));
        });
        element.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'absolute z-50 px-2 py-1 text-sm text-white bg-gray-900 rounded shadow-lg';
    tooltip.textContent = text;
    tooltip.id = 'tooltip';
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
}

function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy to clipboard', 'error');
    });
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 z-50 px-4 py-2 rounded shadow-lg text-white fade-in ${
        type === 'success' ? 'bg-green-500' : 
        type === 'error' ? 'bg-red-500' : 
        'bg-blue-500'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Export functionality
function exportResults(format = 'json') {
    const params = new URLSearchParams(currentFilters);
    params.set('export', format);
    
    window.location.href = '/export?' + params.toString();
}

// Theme toggle
function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-theme');
    
    const isDark = body.classList.contains('dark-theme');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Load saved theme
function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
}

// Collapsible sections
function toggleCollapse(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.toggle('hidden');
        
        // Update icon if present
        const toggle = document.querySelector(`[onclick*="${elementId}"]`);
        if (toggle) {
            const icon = toggle.querySelector('.collapse-icon');
            if (icon) {
                icon.textContent = element.classList.contains('hidden') ? '▶' : '▼';
            }
        }
    }
}

// Auto-refresh functionality
let autoRefreshInterval = null;

function toggleAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        showNotification('Auto-refresh disabled', 'info');
    } else {
        autoRefreshInterval = setInterval(() => {
            // Refresh current tab data
            const activeTab = document.querySelector('.tab-button.active');
            if (activeTab) {
                const tabName = activeTab.id.replace('tab-', '');
                // Only refresh if we're on a data tab
                if (['table', 'errors'].includes(tabName)) {
                    location.reload();
                }
            }
        }, 30000); // Refresh every 30 seconds
        showNotification('Auto-refresh enabled (30s)', 'success');
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    loadTheme();
    initializeTooltips();
    initializeSearch();
    
    // Initialize charts if on overview tab
    if (document.querySelector('#overview-content:not(.hidden)')) {
        setTimeout(initializeCharts, 100);
    }
    
    // Handle file upload with drag and drop
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('bg-blue-50', 'border-blue-300');
        });
        
        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('bg-blue-50', 'border-blue-300');
        });
        
        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('bg-blue-50', 'border-blue-300');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.jsonl')) {
                const fileInput = this.querySelector('input[type="file"]');
                fileInput.files = files;
                
                // Trigger form submission
                const form = this.querySelector('form');
                form.submit();
            }
        });
    }
    
    // Show dashboard content if data is already loaded
    if (window.dataLoaded) {
        document.getElementById('dashboard-content').classList.remove('hidden');
        document.getElementById('upload-section').classList.add('hidden');
    }
});