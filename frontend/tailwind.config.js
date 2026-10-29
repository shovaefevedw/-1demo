/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "SF Pro Text",
          "Avenir Next",
          "Segoe UI",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};
