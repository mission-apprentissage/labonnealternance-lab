module.exports = {
  branches: ["main", { name: "hotfix", channel: "hotfix", prerelease: "hotfix" }],
  plugins: [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    [
      "@semantic-release/exec",
      {
        prepareCmd: `.bin/mna-lab app:release \${nextRelease.version} push`,
      },
    ],
    "@semantic-release/github",
    [
      "semantic-release-slack-bot",
      {
        packageName: "labonnealternance-lab",
        notifyOnSuccess: true,
        notifyOnFail: true,
        markdownReleaseNotes: true,
      },
    ],
  ],
};
