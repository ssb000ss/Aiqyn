/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        bg:              "#0F1117",
        surface:         "#161B27",
        card:            "#1C2333",
        accent:          "#3B82F6",
        "accent-hover":  "#2563EB",
        "text-primary":   "#F1F5F9",
        "text-secondary": "#94A3B8",
        "text-muted":     "#475569",
        border:          "#1E2D45",
        "score-human":  "#22C55E",
        "score-mixed":  "#F59E0B",
        "score-ai":     "#EF4444",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
