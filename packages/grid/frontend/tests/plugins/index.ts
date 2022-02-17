module.exports = (on, config) => {
    config.baseUrl = process.env.NEXT_PUBLIC_HOST
    return config
}