# Sudoku Solver PWA 🎯

A mobile-first Progressive Web App that extracts Sudoku puzzles from photos and solves them using AI.

## ✨ Features

### 📸 Smart Camera Capture
- Full-screen viewfinder with corner bracket guides
- Camera and gallery upload options
- Flash toggle for low-light conditions
- Auto-detection of puzzle grids

### 🤖 AI-Powered Processing
- Simulated OCR grid detection with animated visualization
- Confidence scoring for extracted numbers
- Low-confidence cell warnings

### ✏️ Interactive Grid Review
- Tap-to-edit cells with number picker overlay
- Real-time validation highlighting duplicate errors
- Orange dots mark low-confidence OCR results
- Red highlighting for validation errors

### ⚡ Animated Solving
- Sweeping gradient animation during solve
- Live iteration counter
- Backtracking algorithm visualization

### 🎉 Solution Display
- Original numbers in cyan, solved in green
- Sequential box-by-box animation
- Statistics: solve time, clues count, difficulty
- Share and save options

### 📚 History Management
- Grid view of past solves with miniature puzzles
- Filter by source (Camera/Upload/Manual) or Favorites
- Star favorites for quick access
- Delete unwanted entries
- Difficulty badges and timestamps

### 🔢 Manual Entry
- Empty grid with large number pad
- Auto-advance to next cell
- Minimum 17 clues validation
- Perfect for typing puzzles from books

### 🎨 Design Features
- Glass-morphism buttons with gradient effects
- Smooth Motion animations and transitions
- Responsive mobile-first layout
- Dark theme optimized for OLED screens
- PWA install prompt for home screen

## 🛠️ Technology Stack

- **React** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS v4** - Styling
- **Motion (Framer Motion)** - Animations
- **Lucide React** - Icon library
- **Sonner** - Toast notifications
- **LocalStorage** - History persistence
- **Vite** - Build tool

## 🚀 Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 📱 PWA Features

- Installable on mobile devices
- Offline-capable with service worker
- Home screen icon
- Standalone app experience
- Fast loading and caching

## 🧩 Sudoku Algorithm

Uses a backtracking algorithm to solve puzzles:
1. Find empty cell
2. Try numbers 1-9
3. Check if valid (no duplicates in row/column/box)
4. Recursively solve
5. Backtrack if no solution found

## 📂 Project Structure

```
src/
├── app/
│   ├── components/
│   │   ├── enhanced-sudoku-grid.tsx
│   │   ├── glass-button.tsx
│   │   ├── number-picker.tsx
│   │   └── pwa-install-prompt.tsx
│   ├── screens/
│   │   ├── camera-screen.tsx
│   │   ├── processing-screen.tsx
│   │   ├── grid-review-screen.tsx
│   │   ├── solving-screen.tsx
│   │   ├── solution-screen.tsx
│   │   ├── history-screen.tsx
│   │   ├── manual-entry-screen.tsx
│   │   ├── error-screen.tsx
│   │   └── splash-screen.tsx
│   ├── utils/
│   │   ├── sudoku-solver.ts
│   │   └── storage.ts
│   ├── types/
│   │   └── index.ts
│   └── App.tsx
└── styles/
```

## 🎯 Key Interactions

- **Camera Shutter**: Smooth capture animation with feedback
- **Grid Detection**: Animated cyan lines drawing over detected puzzle
- **Cell Editing**: Bottom sheet number picker with haptic-like feedback
- **Solving Animation**: Sweeping gradient with live counter
- **Solution Reveal**: Sequential fade-in by 3×3 boxes

## 📝 Notes

This is a demo application. The OCR functionality is simulated and returns a sample puzzle. In production, you would integrate a real OCR library like Tesseract.js or a cloud OCR API.

## 🔮 Future Enhancements

- Real OCR integration (Tesseract.js)
- Puzzle difficulty generator
- Hint system
- Step-by-step solution walkthrough
- Multiple solving strategies
- Export to PDF
- Social sharing with images
- Dark/light theme toggle
- Accessibility improvements
- Multi-language support
