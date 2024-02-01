const tailwindcss = require("tailwindcss")
const autoprefixer = require("autoprefixer")

const config = {
  plugins: [tailwindcss(), autoprefixer],
}

module.exports = config
