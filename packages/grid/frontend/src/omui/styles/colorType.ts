type Colors =
  | 'gray'
  | 'blue'
  | 'lime'
  | 'red'
  | 'magenta'
  | 'orange'
  | 'marigold'
  | 'cyan'
  | 'purple'
  | 'violet'
type OmuiThemeColors = 'primary' | 'gray' | 'success' | 'error' | 'warning'

export type OmuiColors = Colors | OmuiThemeColors
