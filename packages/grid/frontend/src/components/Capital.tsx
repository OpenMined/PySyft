import React from 'react'
import tw, { styled } from 'twin.macro'
import CloseIcon from '$icons/CloseIonicons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Button, ButtonGroup } from '$components/Buttons'
import { IconName } from '@fortawesome/free-solid-svg-icons'
import { ThemeMode } from '$types'
import type { PropsWithChildren } from 'react'
import type { ButtonProps } from '$components/Buttons'

export const Container = styled.section`
  ${tw`flex flex-col px-2.5 py-1.5 shadow-card-neutral-1`}
`

interface CapitalProps {
  title?: string
  leftButtons: Array<ButtonProps>
  rightButtons: Array<ButtonProps>
  icon: IconName
  mode?: ThemeMode
  type: 'blank' | 'center' | 'left'
  close?: () => void
}

type ExtendedCapitalProps = {
  isBlank?: boolean
  isCentered?: boolean
  hasIcon?: boolean
} & PropsWithChildren<CapitalProps>

export const Capital = ({
  title,
  leftButtons,
  rightButtons,
  icon,
  mode,
  type,
  close,
  children,
}: PropsWithChildren<CapitalProps>) => {
  const isBlank = type === 'blank'
  const isCentered = type === 'center'
  const hasIcon = Boolean(icon)

  const headerProps = { title, mode, isBlank, isCentered, hasIcon, close }
  const bodyProps = { children, mode, isCentered }
  const footerProps = { mode, leftButtons, rightButtons }

  return (
    <Container>
      <Header {...headerProps} />
      <Body {...bodyProps}>{children}</Body>
      <Footer {...footerProps} />
    </Container>
  )
}

export const HeaderBox = styled.header(
  ({ hasIcon, mode, isBlank, isCentered }: Partial<ExtendedCapitalProps>) => [
    tw`flex min-h-15 px-6`,
    isBlank ? tw`py-4 justify-end` : tw`pb-6 pt-4`,
    mode === 'dark' && tw`text-gray-0`,
    mode === 'light' && tw`text-gray-800`,
    hasIcon ? tw`flex-col space-y-2` : tw`space-x-2`,
    isCentered && tw`items-center text-center`,
  ]
)

export const FooterBox = styled.footer(({ isCentered }: Partial<ExtendedCapitalProps>) => [
  tw`px-6 pt-6 pb-4 flex`,
  isCentered && tw`justify-center gap-6`,
  !isCentered && tw`justify-between gap-6`,
])

const Title = styled.p`
  ${tw`text-xl font-medium w-full`}
`

const CloseButton = styled.button`
  ${tw`h-6 w-6 items-start flex-shrink-0 self-end`}
`

const Close = ({ close }) => (
  <CloseButton onClick={close}>
    <CloseIcon />
  </CloseButton>
)

const Header = ({
  mode,
  isBlank,
  isCentered,
  title,
  icon,
  close,
}: Partial<ExtendedCapitalProps>) => {
  if (isBlank) {
    return (
      <HeaderBox mode={mode} isBlank>
        <Close close={close} />
      </HeaderBox>
    )
  }

  return (
    <HeaderBox mode={mode} isCentered={isCentered} hasIcon={Boolean(icon)}>
      {!icon && <Title>{title}</Title>}
      <Close close={close} />
      {icon && (
        <>
          <FontAwesomeIcon icon={icon} tw="h-6 w-6" />
          <Title as="h4">{title}</Title>
        </>
      )}
    </HeaderBox>
  )
}

export const Body = styled.section(({ mode, isCentered }: Partial<ExtendedCapitalProps>) => [
  tw`px-6 pb-3`,
  mode === 'dark' && tw`text-gray-200`,
  mode === 'light' && tw`text-gray-600`,
  isCentered && tw`text-center`,
])

const Footer = ({
  mode = 'light',
  isCentered,
  leftButtons = [],
  rightButtons = [],
}: Partial<ExtendedCapitalProps>) => (
  <FooterBox mode={mode} isCentered={isCentered}>
    <ButtonGroup>
      {leftButtons.map(buttonProps => (
        <Button key={buttonProps.copy} {...buttonProps} />
      ))}
    </ButtonGroup>
    <ButtonGroup>
      {rightButtons.map(buttonProps => (
        <Button key={buttonProps.copy} {...buttonProps} />
      ))}
    </ButtonGroup>
  </FooterBox>
)
