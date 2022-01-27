import { Global, css } from '@emotion/react'
import tw, { GlobalStyles as BaseStyles } from 'twin.macro'

const customStyles = css`
  * {
    padding: 0;
    margin: 0;
    box-sizing: border-box;
  }

  html,
  body,
  body > div:first-child,
  div#__next,
  div#__next > div,
  main {
    ${tw`antialiased h-full`}
    ${tw`font-roboto text-base leading-[1.5]`}
    ${tw`text-gray-800`}
  }

  h1,
  h2,
  h3,
  h4,
  h5 {
    font-family: 'Rubik';
    ${tw`!font-medium`}
  }

  h1 {
    ${tw`(text-5xl leading-[1.1])!`}
  }

  h2 {
    ${tw`(text-4xl leading-[1.1])!`}
  }

  h3 {
    ${tw`(text-3xl leading-[1.4])!`}
  }

  h4 {
    ${tw`(text-2xl leading-[1.5])!`}
  }

  h5 {
    ${tw`(text-xl leading-[1.5])!`}
  }

  h6 {
    ${tw`(text-sm leading-[1.5] font-bold)!`}
  }

  a {
    ${tw`(cursor-pointer text-primary-600 underline hover:text-blue-400 active:text-blue-300)!`}
  }
`

export const GlobalStyles = () => (
  <>
    <BaseStyles />
    <Global styles={customStyles} />
  </>
)
