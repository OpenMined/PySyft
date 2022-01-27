import { theme } from 'twin.macro'

interface CloseIconProps {
  color?: string
}

// #736C93
const CloseIcon = ({ color = theme`textColor.current` }: CloseIconProps) => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path
      d="M17.2501 17.25L6.75006 6.75"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />

    <path
      d="M17.2501 6.75L6.75006 17.25"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

export default CloseIcon
