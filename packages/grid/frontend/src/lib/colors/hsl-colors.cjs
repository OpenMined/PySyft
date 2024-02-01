function buildColor(colorName, hue, saturation, baseLightness) {
  const MIN_L = 3
  const MAX_L = 95

  const baseColorLightness = baseLightness || 50

  if ((baseLightness > 50 || baseLightness < 40) && colorName != "white") {
    throw new Error(`baseLightness must be between 40 and 50`)
  }

  const maxColorLightness = Math.max(baseColorLightness - 40, MIN_L)
  const minColorLightness = Math.min(baseColorLightness + 50, MAX_L)

  return {
    [colorName]: {
      50: `hsl(${hue}, ${saturation}%, ${minColorLightness}%)`,
      100: `hsl(${hue}, ${saturation}%, ${baseColorLightness + 40}%)`,
      200: `hsl(${hue}, ${saturation}%, ${baseColorLightness + 30}%)`,
      300: `hsl(${hue}, ${saturation}%, ${baseColorLightness + 20}%)`,
      400: `hsl(${hue}, ${saturation}%, ${baseColorLightness + 10}%)`,
      500: `hsl(${hue}, ${saturation}%, ${baseColorLightness}%)`,
      600: `hsl(${hue}, ${saturation}%, ${baseColorLightness - 10}%)`,
      700: `hsl(${hue}, ${saturation}%, ${baseColorLightness - 20}%)`,
      800: `hsl(${hue}, ${saturation}%, ${baseColorLightness - 30}%)`,
      900: `hsl(${hue}, ${saturation}%, ${maxColorLightness}%)`,
    },
  }
}

module.exports = {
  ...buildColor("blue", 225, 70, 50),
  ...buildColor("cyan", 195, 80, 50),
  ...buildColor("gray", 252, 15, 50),
  ...buildColor("lime", 120, 54, 43),
  ...buildColor("magenta", 330, 75, 50),
  ...buildColor("marigold", 50, 85, 50),
  ...buildColor("orange", 20, 85, 50),
  ...buildColor("purple", 255, 65, 50),
  ...buildColor("red", 350, 75, 50),
  ...buildColor("violet", 280, 64, 43),
  ...buildColor("green", 120, 54, 43),
  ...buildColor("black", 251, 16, 43),
  ...buildColor("white", 0, 0, 0),
}
