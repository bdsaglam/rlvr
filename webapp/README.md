# Multi-Hop QA Results Inspector

A FastHTML web application for interactive error analysis and trajectory inspection of multi-hop question answering evaluation results.

## Features

### 🎯 **Dashboard Overview**
- Summary statistics and metrics breakdown
- Performance analysis by number of hops
- Reward component distributions
- Error rate visualizations

### 📊 **Interactive Results Table**
- Sortable and filterable results
- Quick status indicators (✅❌)
- Filter by correctness, hop count, reward thresholds
- Direct links to detailed trajectory views

### 🔍 **Detailed Trajectory View**
- Full conversation flow with clear role separation
- Tool call visualization with arguments and responses
- Document usage tracking with supporting doc highlights
- Citation analysis and format compliance details
- Comprehensive metrics breakdown

### 🔬 **Error Analysis Tools**
- Reward component comparison (correct vs incorrect)
- Pattern analysis by failure modes
- Supporting document usage statistics
- Tool call pattern analysis
- Quick identification of problem areas

### 🎨 **Great UX Features**
- Responsive design for desktop and mobile
- Keyboard shortcuts (Alt+1/2/3 for tabs, Ctrl/Cmd+K for search)
- Drag-and-drop file upload
- Real-time filtering and search
- Export functionality
- Auto-refresh capability
- Tooltips and notifications

## Usage

### Quick Start
```bash
# Navigate to the webapp directory
cd webapp

# Run the application (uv will automatically install dependencies)
uv run app.py
```

The app will start on http://localhost:8000

### Loading Data
1. Visit the web interface
2. Drag and drop your `.jsonl` evaluation results file, or use the file picker
3. The dashboard will automatically load with your data

### Navigation
- **Overview Tab**: Summary statistics and performance breakdown
- **Results Table Tab**: Filterable table of all examples
- **Error Analysis Tab**: Detailed error pattern analysis

### Filtering
- Use the filter buttons to show only correct/incorrect examples
- Filter by number of hops using the dropdown
- Set minimum reward thresholds
- Filters apply across all views

### Keyboard Shortcuts
- `Alt + 1`: Overview tab
- `Alt + 2`: Results table tab  
- `Alt + 3`: Error analysis tab
- `Ctrl/Cmd + K`: Focus search (when implemented)

## Data Format

The app expects JSONL files where each line contains an evaluation result with this structure:

```json
{
  "prompt": [...],                    // Conversation prompt
  "completion": [...],                // Full conversation trajectory
  "answer": "predicted answer",       // Model's final answer
  "info": {
    "answers": ["correct", "answer"], // Reference answers
    "n_hops": 2,                     // Number of reasoning hops
    "docs": [...]                    // Available documents
  },
  "task": "default",
  "reward": 0.367,                   // Overall reward
  "exact_match_reward": 0.0,         // Exact match component
  "f1_reward": 0.0,                  // F1 score component
  "retrieval_recall_reward": 1.0,    // Retrieval quality
  "citation_reward": 1.0,            // Citation quality
  "format_reward": 1.0,              // Format compliance
  "combined_reward": 0.367           // Combined score
}
```

## Dependencies

All dependencies are automatically managed via uv script dependencies:

- `fasthtml>=0.3.0` - Web framework
- `pandas>=2.0.0` - Data processing  
- `uvicorn>=0.24.0` - ASGI server
- `python-multipart>=0.0.6` - File upload support

## Architecture

```
webapp/
├── app.py              # Main FastHTML application
├── data_loader.py      # Data loading and processing utilities  
├── components.py       # Reusable UI components
├── static/
│   ├── style.css      # Custom styling
│   └── app.js         # Interactive JavaScript
└── README.md          # This file
```

## Customization

### Adding New Metrics
1. Update `data_loader.py` to extract new metrics from your data
2. Add visualization in `components.py`
3. Update the dashboard display logic in `app.py`

### Styling
- Modify `static/style.css` for visual customizations
- The app uses Tailwind CSS classes throughout
- Custom components can be styled with additional CSS classes

### New Analysis Views
1. Create new component functions in `components.py`
2. Add route handlers in `app.py`
3. Add navigation elements and JavaScript in `static/app.js`

## Production Deployment

For production use:

1. Set up a proper database instead of in-memory storage
2. Add authentication and session management
3. Configure reverse proxy (nginx) and SSL
4. Set up monitoring and logging
5. Use production ASGI server configuration

## Troubleshooting

- **File not loading**: Ensure your file is valid JSONL format
- **Missing metrics**: Check that your data includes the expected reward columns
- **Performance issues**: For large datasets, consider pagination or data sampling
- **Display issues**: Clear browser cache and ensure JavaScript is enabled

## Contributing

The codebase follows clean architecture principles:
- Data processing is separated from presentation
- Components are reusable and modular
- JavaScript functionality is organized by feature
- Styling uses utility-first CSS approach