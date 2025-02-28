/** @type {import('tailwindcss').Config} */
import typography from "@tailwindcss/typography";
import daisyui from "daisyui";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      // Removing any custom color definitions to rely on DaisyUI theme
    },
  },
  plugins: [typography, daisyui],
  daisyui: {
    themes: ["lofi"], // You can change this to any other DaisyUI theme if needed
    base: true,
    styled: true,
    utils: true,
    prefix: "",
    logs: true,
    themeRoot: ":root",
  },
};
