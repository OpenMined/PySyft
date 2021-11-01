export function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']

  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

export function localeSortByVariable<T>(array: T[], variable: string): T[] {
  return array.sort((a, b) => a[variable].localeCompare(b[variable]))
}

export function singularOrPlural(singular = '', plural = '', value = null) {
  if (value === 1) return singular
  return plural
}

export function formatBudget(budget: number | string): string {
  if (!budget) return '0.00'
  return Number(budget).toFixed(2)
}
