function buildColor(name, hue, saturation) {
  // const colorScales = [{code: 50, lightness: 95}, {code: 100, lightness: 90}]
  const lightnessScales = [95, 90, 80, 70, 60, 43, 33, 23, 13, 3]
  const colorCodes = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
  const mapping = lightnessScales.map(
    (lightness) => `hsl(${hue}, ${saturation}%, ${lightness}%)`
  )
  const colorObject = {}

  colorCodes.forEach((code, index) => {
    colorObject[code] = mapping[index]
  })

  return {
    [name]: colorObject,
  }
}

module.exports = {
  ...buildColor('gray', 252, 15),
  ...buildColor('red', 350, 75),
  ...buildColor('magenta', 330, 75),
  ...buildColor('orange', 20, 85),
  ...buildColor('marigold', 37, 85),
  ...buildColor('lime', 120, 54),
  ...buildColor('cyan', 195, 75),
  ...buildColor('blue', 225, 70),
  ...buildColor('purple', 255, 65),
  ...buildColor('violet', 280, 64),
}
