/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#070a0f",
          panel: "#0d121b",
          panel2: "#111826",
          border: "#1c2533",
          borderlit: "#2a3647",
          text: "#d6dde8",
          muted: "#7c8aa0",
          dim: "#56627a",
        },
        bull: "#13c98a",
        bear: "#f0445a",
        side: "#e0a83b",
        accent: "#3fa7ff",
        violet: "#9b7bff",
      },
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "ui-monospace",
          "monospace",
        ],
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 24px -12px rgba(0,0,0,0.6)",
      },
    },
  },
  plugins: [],
};
