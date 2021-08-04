import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
import localizedFormat from 'dayjs/plugin/localizedFormat'
import type {Dayjs} from 'dayjs'

dayjs.extend(relativeTime)
dayjs.extend(localizedFormat)

type DateArgs = string | number | Date | Dayjs

export function format(date: DateArgs): string {
  return dayjs(date).format('l')
}

export function formatFullDate(date: DateArgs): string {
  return dayjs(date).format('llll')
}

export function dateFromNow(date: DateArgs): string {
  return dayjs(date).fromNow()
}
