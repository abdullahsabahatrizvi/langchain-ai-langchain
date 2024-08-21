/* eslint-disable global-require,import/no-extraneous-dependencies */

// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
// eslint-disable-next-line import/no-extraneous-dependencies
const { ProvidePlugin } = require("webpack");
require("dotenv").config();

const baseLightCodeBlockTheme = require("prism-react-renderer/themes/vsLight");
const baseDarkCodeBlockTheme = require("prism-react-renderer/themes/vsDark");

const baseUrl = "/v0.2/";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "🦜️🔗 LangChain",
  tagline: "LangChain Python Docs",
  favicon: "img/brand/favicon.png",
  // Set the production url of your site here
  url: "https://python.langchain.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: baseUrl,
  trailingSlash: true,
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "throw",

  themes: ["@docusaurus/theme-mermaid"],
  markdown: {
    mermaid: true,
  },

  plugins: [
    () => ({
      name: "custom-webpack-config",
      configureWebpack: () => ({
        plugins: [
          new ProvidePlugin({
            process: require.resolve("process/browser"),
          }),
        ],
        resolve: {
          fallback: {
            path: false,
            url: false,
          },
        },
        module: {
          rules: [
            {
              test: /\.m?js/,
              resolve: {
                fullySpecified: false,
              },
            },
            {
              test: /\.py$/,
              loader: "raw-loader",
              resolve: {
                fullySpecified: false,
              },
            },
            {
              test: /\.ya?ml$/,
              use: 'yaml-loader'
            },
            {
              test: /\.ipynb$/,
              loader: "raw-loader",
              resolve: {
                fullySpecified: false,
              },
            },
          ],
        },
      }),
    }),
  ],

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          editUrl: "https://github.com/langchain-ai/langchain/edit/master/docs/",
          sidebarPath: require.resolve("./sidebars.js"),
          remarkPlugins: [
            [require("@docusaurus/remark-plugin-npm2yarn"), { sync: true }],
          ],
          async sidebarItemsGenerator({
            defaultSidebarItemsGenerator,
            ...args
          }) {
            const sidebarItems = await defaultSidebarItemsGenerator(args);
            sidebarItems.forEach((subItem) => {
              // This allows breaking long sidebar labels into multiple lines
              // by inserting a zero-width space after each slash.
              if (
                "label" in subItem &&
                subItem.label &&
                subItem.label.includes("/")
              ) {
                // eslint-disable-next-line no-param-reassign
                subItem.label = subItem.label.replace(/\//g, "/\u200B");
              }
              if (args.item.className) {
                subItem.className = args.item.className;
              }
            });
            return sidebarItems;
          },
        },
        pages: {
          remarkPlugins: [require("@docusaurus/remark-plugin-npm2yarn")],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      announcementBar: {
        content: 'LangChain 0.2 is out! Leave feedback on the v0.2 docs <a href="https://github.com/langchain-ai/langchain/discussions/21716">here</a>. You can view the v0.1 docs <a href="/v0.1/docs/get_started/introduction/">here</a>.',
        isCloseable: true,
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: true,
        },
      },
      colorMode: {
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      prism: {
        theme: {
          ...baseLightCodeBlockTheme,
          plain: {
            ...baseLightCodeBlockTheme.plain,
            backgroundColor: "#F5F5F5",
          },
        },
        darkTheme: {
          ...baseDarkCodeBlockTheme,
          plain: {
            ...baseDarkCodeBlockTheme.plain,
            backgroundColor: "#222222",
          },
        },
      },
      image: "img/brand/theme-image.png",
      navbar: {
        logo: {src: "img/brand/wordmark.png", srcDark: "img/brand/wordmark-dark.png"},
        items: [
          {
            type: "docSidebar",
            position: "left",
            sidebarId: "integrations",
            label: "Integrations",
          },
          {
            type: "dropdown",
            label: "API reference",
            position: "left",
            items: [
              {
                label: "Latest",
                to: "https://python.langchain.com/v0.2/api_reference/reference.html",
              },
              {
                label: "Legacy",
                href: "https://api.python.langchain.com/"
              }
            ]
          },
          {
            type: "dropdown",
            label: "More",
            position: "left",
            items: [
              {
                type: "doc",
                docId: "contributing/index",
                label: "Contributing",
              },
              {
                label: "Cookbooks",
                href: "https://github.com/langchain-ai/langchain/blob/master/cookbook/README.md"
              },
              {
                type: "doc",
                docId: "people",
                label: "People",
              },
            ]
          },
          {
            type: "dropdown",
            label: "v0.2",
            position: "right",
            items: [
              {
                label: "v0.2",
                href: "/docs/introduction"
              },
              {
                label: "v0.1",
                href: "https://python.langchain.com/v0.1/docs/get_started/introduction"
              }
            ]
          },
          {
            to: "https://chat.langchain.com",
            label: "💬",
            position: "right",
          },
          // Please keep GitHub link to the right for consistency.
          {
            href: "https://github.com/langchain-ai/langchain",
            position: "right",
            className: "header-github-link",
            "aria-label": "GitHub repository",
          },
        ],
      },
      footer: {
        style: "light",
        links: [
          {
            title: "Ecosystem",
            items: [
              {
                label: "LangChain JS",
                to: "https://js.langchain.com/",
              },
              {
                label: "LangGraph",
                to: "https://langchain-ai.github.io/langgraph/",
              },
              {
                label: "LangSmith",
                to: "https://docs.smith.langchain.com/",
              },
            ],
          },
          {
            title: "Resources",
            items: [
              {
                label: "GitHub",
                to: "https://github.com/langchain-ai",
              },
              {
                label: "Homepage",
                to: "https://langchain.com",
              },
              {
                label: "X / Twitter",
                to: "https://twitter.com/LangChainAI",
              },
              {
                label: "Blog",
                to: "https://blog.langchain.dev",
              },
              {
                label: "YouTube",
                to: "https://www.youtube.com/@LangChain",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} LangChain, Inc.`,
      },
      algolia: {
        // The application ID provided by Algolia
        appId: "VAU016LAWS",

        // Public API key: it is safe to commit it
        // this is linked to erick@langchain.dev currently
        apiKey: "6c01842d6a88772ed2236b9c85806441",

        indexName: "python-langchain-0.2",

        contextualSearch: false,
      },
    }),

  scripts: [
    baseUrl + "js/google_analytics.js",
    {
      src: "https://www.googletagmanager.com/gtag/js?id=G-9B66JQQH2F",
      async: true,
    },
  ],

  customFields: {
    supabasePublicKey: process.env.NEXT_PUBLIC_SUPABASE_PUBLIC_KEY,
    supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL,
  },
};

module.exports = config;
