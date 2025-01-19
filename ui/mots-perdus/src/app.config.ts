import { environment } from "./environments/environment";

export const AppConfig = {
  // API base URL
  apiUrl: environment.apiUrl,

  // App-specific settings
  appTitle: 'La Recherche du Mots Perdus',
  defaultLanguage: 'en',

  // Feature flags
  featureFlags: {
    enableFileUpload: true,
    enableSearch: true,
  },

  // Optionally add environment-specific variables
  environment: environment.production ? 'production' : 'development',
};
