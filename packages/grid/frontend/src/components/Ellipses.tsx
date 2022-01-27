import 'twin.macro'

const GaussianBlur = ({ id, x, y, width, height, stdDeviation }) => (
  <filter
    id={id}
    x={x}
    y={y}
    width={width}
    height={height}
    filterUnits="userSpaceOnUse"
    colorInterpolationFilters="sRGB"
  >
    <feFlood floodOpacity="0" result="BackgroundImageFix" />
    <feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
    <feGaussianBlur stdDeviation={stdDeviation} result="foreground_blur" />
  </filter>
)

const LinearGradient = ({ id, x1, y1, x2, y2 }) => (
  <linearGradient id={id} x1={x1} y1={y1} x2={x2} y2={y2} gradientUnits="userSpaceOnUse">
    <stop stopColor="white" stopOpacity="0.5" />
    <stop offset="1" stopColor="white" stopOpacity="0" />
  </linearGradient>
)

const Circle = ({ gaussianId, cx, cy, r, fill, linearId }) => (
  <g filter={`url(#${gaussianId})`}>
    <circle cx={cx} cy={cy} r={r} fill={fill} />
    <circle cx={cx} cy={cy} r={r} fill={`url(#${linearId})`} />
  </g>
)

const Ellipse = ({ gaussianId, cx, cy, rx, ry, fill, linearId }) => (
  <g filter={`url(#${gaussianId})`}>
    <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill={fill} />
    <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill={`url(#${linearId})`} />
  </g>
)

export const Ellipses = () => (
  <svg
    preserveAspectRatio="xMinYMin slice"
    width="100%"
    height="100%"
    tw="max-h-screen"
    viewBox="0 0 816 970"
    css={['overflow: visible']}
  >
    <Circle
      cx="785.105"
      cy="109.895"
      r="122.895"
      fill="#20AFDF"
      linearId="top-gradient"
      gaussianId="top"
    />
    <Ellipse
      cx="542.846"
      cy="426.315"
      rx="365.154"
      ry="389.874"
      fill="#EB4913"
      linearId="middle-gradient"
      gaussianId="middle"
    />
    <Circle
      cx="504"
      cy="465.867"
      r="404"
      fill="#EC9913"
      linearId="bottom-gradient"
      gaussianId="bottom"
    />
    <defs>
      <GaussianBlur id="top" x={592.21} y={-83} width={385.79} height={385.79} stdDeviation={35} />
      <GaussianBlur
        id="middle"
        x={107.692}
        y={-33.5594}
        width={870.308}
        height={919.748}
        stdDeviation={35}
      />
      <GaussianBlur id="bottom" x={0} y={-38.1329} width={1008} height={1008} stdDeviation={50} />
      <LinearGradient id="top-gradient" x1={662.21} y1={113.859} x2={908} y2={113.859} />
      <LinearGradient id="middle-gradient" x1={177.692} y1={438.891} x2={908} y2={438.891} />
      <LinearGradient id="bottom-gradient" x1={100} y1={478.899} x2={908} y2={478.899} />
    </defs>
  </svg>
)
