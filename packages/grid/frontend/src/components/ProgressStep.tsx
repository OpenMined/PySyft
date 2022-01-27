import tw, { styled } from 'twin.macro'
import { ThemeMode } from '$types'

interface ProgressProps {
  name?: string
  steps: number
  completed: number
  mode?: ThemeMode
}

type StepProps = Partial<ProgressProps> & { isCompleted: boolean }

const Step = styled.div(({ mode = 'light', isCompleted = false }: StepProps) => [
  tw`w-full h-1 rounded-full transition`,
  background[isCompleted ? 'complete' : 'incomplete'][mode],
])

const Container = styled.div`
  ${tw`flex w-full h-5 items-center justify-evenly space-x-1`}
`

export const ProgressStep = ({ name = 'progress', steps = 1, completed = 0 }: ProgressProps) => (
  <Container>
    {Array.from({ length: steps }).map((_, index) => (
      <Step key={`${name}-${index}`} isCompleted={index < completed} />
    ))}
  </Container>
)

const background = {
  complete: {
    dark: tw`bg-primary-400`,
    light: tw`bg-primary-500`,
  },
  incomplete: {
    dark: tw`bg-gray-200`,
    light: tw`bg-gray-200`,
  },
}
