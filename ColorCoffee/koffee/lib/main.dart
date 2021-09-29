import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:koffee/view/splash_screen/splash_screen_page.dart';

import 'theme/theme.dart';

void configLoading() {
  EasyLoading.instance
    ..loadingStyle = EasyLoadingStyle.light
    ..backgroundColor = Colors.white
    ..maskType = EasyLoadingMaskType.black
    ..userInteractions = false
    ..dismissOnTap = false;
}

void main() {
  runApp(const MyApp());
  configLoading();
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // MultiProvider(
    //   providers: [
    //     ChangeNotifierProvider<ThemeChanger>(
    //         create: (_) => ThemeChanger(koffeTheme)),
    //   ],

    return MaterialApp(
      title: 'Flutter Demo',
      theme: AppTheme.koffeTheme,
      home: const SplashScreen(),
      builder: EasyLoading.init(),
    );
  }
}