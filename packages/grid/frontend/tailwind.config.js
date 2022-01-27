module.exports = {
  mode: 'jit',
  purge: ['./src/pages/**/*.{js,ts,jsx,tsx}', './src/components/**/*.{js,ts,jsx,tsx}'],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      fontFamily: {
        roboto: ['Roboto', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
        rubik: ['Rubik', 'sans-serif'],
      },
      backgroundColor: {
        'layout-white': '#f1f0f4',
      },
      minHeight: {
        15: '3.25rem',
      },
      height: {
        15: '3.25rem',
      },
      borderWidth: {
        1.5: '1.5px',
      },
      dropShadow: {
        hover: '-4px 4px 16px #20AFDF',
        'hover-danger': '-4px 4px 16px #DE207F',
      },
      boxShadow: {
        focus: '0px 0px 8px #20AFDF',
        'focus-danger': '0px 0px 8px #DE207F',
        'domain-menu': '0px 0px 5px rgba(113, 128, 150, 0.25)',
        'card-neutral-1': '-2px 4px 8px rgba(13, 12, 17, 0.25)',
      },
      backgroundImage: {
        scrim: {
          light: 'linear-gradient(90deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 100%)',
          dark: 'linear-gradient(90deg, rgba(0, 0, 0, 0.75) 0%, rgba(0, 0, 0, 0) 100%)',
          'gray-dark': 'linear-gradient(90deg, rgba(0, 0, 0, 0.75) 0%, rgba(0, 0, 0, 0.1) 100%)',
          'layout-white':
            'linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%)',
        },
      },
      colors: {
        primary: {
          95: 'hsla(195, 100%, 95%, 1)',
          100: 'hsla(195, 75%, 90%, 1)',
          200: 'hsla(195, 75%, 80%, 1)',
          300: 'hsla(195, 75%, 70%, 1)',
          400: 'hsla(195, 75%, 60%, 1)',
          500: 'hsla(195, 75%, 50%, 1)',
          600: 'hsla(195, 75%, 40%, 1)',
          700: 'hsla(195, 75%, 30%, 1)',
          800: 'hsla(195, 75%, 20%, 1)',
          900: 'hsla(195, 75%, 10%, 1)',
        },
        gray: {
          0: '#fff',
          50: 'hsla(252, 15%, 95%, 1)',
          100: 'hsla(252, 15%, 90%, 1)',
          200: 'hsla(252, 15%, 80%, 1)',
          300: 'hsla(252, 15%, 70%, 1)',
          400: 'hsla(252, 15%, 60%, 1)',
          500: 'hsla(252, 15%, 50%, 1)',
          600: 'hsla(252, 15%, 40%, 1)',
          700: 'hsla(252, 15%, 30%, 1)',
          800: 'hsla(252, 15%, 20%, 1)',
          900: 'hsla(252, 15%, 10%, 1)',
        },
        danger: {
          95: 'hsla(330, 75%, 95%, 1)',
          100: 'hsla(330, 75%, 90%, 1)',
          200: 'hsla(330, 75%, 80%, 1)',
          300: 'hsla(330, 75%, 70%, 1)',
          400: 'hsla(330, 75%, 60%, 1)',
          500: 'hsla(330, 75%, 50%, 1)',
          600: 'hsla(330, 75%, 40%, 1)',
          700: 'hsla(330, 75%, 30%, 1)',
          800: 'hsla(330, 75%, 20%, 1)',
          900: 'hsla(330, 75%, 10%, 1)',
        },
        warning: {
          50: 'hsla(37, 85%, 95%, 1)',
          100: 'hsla(37, 85%, 90%, 1)',
          200: 'hsla(37, 85%, 80%, 1)',
          300: 'hsla(37, 85%, 70%, 1)',
          400: 'hsla(37, 85%, 60%, 1)',
          500: 'hsla(37, 85%, 50%, 1)',
          600: 'hsla(37, 85%, 40%, 1)',
          700: 'hsla(37, 85%, 30%, 1)',
          800: 'hsla(37, 85%, 20%, 1)',
          900: 'hsla(37, 85%, 10%, 1)',
        },
        success: {
          50: 'hsla(120, 54%, 93%, 1)',
          100: 'hsla(120, 54%, 83%, 1)',
          200: 'hsla(120, 54%, 73%, 1)',
          300: 'hsla(120, 54%, 63%, 1)',
          400: 'hsla(120, 54%, 53%, 1)',
          500: 'hsla(120, 54%, 43%, 1)',
          600: 'hsla(120, 54%, 33%, 1)',
          700: 'hsla(120, 54%, 23%, 1)',
          800: 'hsla(120, 54%, 13%, 1)',
          900: 'hsla(120, 54%, 3%, 1)',
        },
      },
    },
  },
  variants: {
    extend: {
      backgroundImage: ['responsive', 'hover', 'focus', 'group-hover'],
      opacity: ['responsive', 'hover', 'focus', 'group-hover', 'disabled'],
      ringWidth: ['responsive', 'hover', 'focus', 'group-hover', 'focus-visible'],
    },
  },
  plugins: [],
}
