const openminedColors = require("./src/lib/colors/hsl-colors.cjs")

module.exports = {
  content: ["./src/**/*.{html,js,svelte,ts}"],
  theme: {
    extend: {
      fontFamily: {
        roboto: ["Roboto", "Roboto-offline", "system-ui", "sans-serif"],
        rubik: ["Rubik", "Rubik-offline", "system-ui", "sans-serif"],
        fira: ["Fira Code", "Fira Code-offline", "ui-monospace", "monospace"],
        mono: ["Fira Code", "Fira Code-offline", "ui-monospace", "monospace"],
      },
      colors: {
        ...openminedColors,
        primary: openminedColors.cyan,
        white: "white",
        black: "black",
        pressedGray: "hsla(255, 15%, 95%, 0.45)",
      },
      spacing: {
        4.5: "1.125rem",
        13: "3.25rem",
      },
      lineHeight: {
        120: "1.2",
        160: "1.6",
      },
      boxShadow: {
        "neutral-1": "-2px 4px 8px rgba(13, 12, 17, 0.25)",
        "modal-1": "-1px 2px 4px 2px rgba(0, 0, 0, 0.08)",
        "tooltip-1": "0px 1px 2px 0px rgba(0, 0, 0, 0.25)",
        "topbar-1": "0px 3px 16px 0px rgba(0, 0, 0, 0.1)",
        "callout-1": "0px 1px 2px 1px rgba(0, 0, 0, 0.05)",
        "roles-1": "0px 2px 2px 2px rgba(0, 0, 0, 0.05)",
      },
      minHeight: {
        screen: "100vh",
      },
      minWidth: {
        screen: "100vw",
      },
      screens: {
        tablet: "700px",
        desktop: "1280px",
      },
    },
  },
  plugins: [],
  darkMode: "class",
}
