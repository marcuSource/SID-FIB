
//TEAM_AXIS (Equipo 200)

//PLAN 1 (ANAR A LA BANDERA I PATRULLAR) Identificat amb PATROLLING
+flag(F): team(200) 
  <-
  .print("Axis soldier: Voy hacia la bandera");
  +anar_a_bandera;
  .goto(F).

+target_reached(T): anar_a_bandera & team(200)
  <-
  .print("Axis soldier: Llegué a la bandera, creando puntos de patrulla");
  -anar_a_bandera;
  -target_reached(T);
  ?flag([FX,FY,FZ]); // Posición de la bandera
  
  // Crear 4 puntos en forma de rombo alrededor de la bandera a 10 unidades
  P1 = [FX, FY, FZ+10];    // Norte
  P2 = [FX+10, FY, FZ];    // Este
  P3 = [FX, FY, FZ-10];    // Sur
  P4 = [FX-10, FY, FZ];    // Oeste
  
  // Los ordenamos en sentido antihorario
  C = [P2, P1, P4, P3];
  
  +control_points(C);
  .length(C,L);
  +total_control_points(L);
  +patrullant;
  +patroll_point(0);
  //?control_points(C);
  .nth(0,C,A);
  .goto(A).

+target_reached(T): patrullant & team(200)
  <-
  .print("Axis soldier: Punto de patrulla alcanzado");
  ?patroll_point(P);
  -+patroll_point(P+1);
  -target_reached(T).

+patroll_point(P): total_control_points(T) & P<T & team(200)
  <-
  ?control_points(C);
  .nth(P,C,A);
  .goto(A).

+patroll_point(P): total_control_points(T) & P==T & team(200)
  <-
  -patroll_point(P);
  +patroll_point(0).


//-----------------FINAL DEL COMPORTAMENT GENERAL, ARA ACCIONS ESPECÍFIQUES -------------------------

//PLAN 1 PATRULLANT
// Si trobem enemics disparem i bloquejem el plantejament a combatre
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): Health > 0 & patrullant & team(200)
  <-
  .look_at([X1,Y1,Z1]);
  .shoot(5,[X1,Y1,Z1]);
  -patrullant;
  +combatent;
  .print("Axis soldier: Disparo al enemigo");
  .stop.


// Si tenim poca vida busquem medikits
+health(H): team(200) & H < 50 & patrullant
  <-
  .print("Axis soldier: Vida baja, buscando medkit");
  .stop;
  .turn(0.25).


//PLAN 2 COMBATENT

+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): Health > 0 & combatent & team(200)
  <-
  .shoot(5,[X1,Y1,Z1]);
  .print("Axis soldier: Disparo al enemigo (después de recibir daño)");
  .stop;
  .look_at([X1,Y1,Z1]).

//hem deixat de veure enemics mentre combatiem tornem a patrullar
+combatent: not enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]) & team(200)
  <-
  ?control_points(C);
  -combatent;
  +patrullant;
  .nth(0,C,A);
  .goto(A).

//Si nomes tinc 20 de vida o menys deixo de combatre i fujo per anar a la bandera on sé que hi ha medikits (Al arribar a la bandera tornaré a la posició on estava)
+health(X): X < 20 & combatent & team(200)
  <-
  -combatent;
  +fujint_per_tornar;
  ?position(P);
  +posicio_abans_de_fujir(P);
  ?flag(F);
  .look_at(F);
  .goto(F).

+target_reached(T): team(200) & fujint_per_tornar
  <-
  -target_reached(T);
  -fujint_per_tornar;
  .print("Axis soldier: Tornem a combatre al lloc on erem");
  +combatent;
  ?posicio_abans_de_fujir(P);
  .goto(P);
  .look_at(P).

//PLA 3 PERSEGUINT:
//Veiem que algú s'emporta la bandera
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & flag(F) & (F < Position | F > Position) & Type == 1003
  <-
  .print("HE VIST LA BANDERA, la perseguim");
  -patrullant;
  +perseguint_bandera;
  .look_at(Position);
  .goto(Position).

+target_reached(T) : team(200) & perseguint_bandera
  <-
  .print("Hem arribat a la bandera, ja no la tindra l'enemic, patrullo al voltant");
  -target_reached(T);
  -perseguint_bandera;
  +patrullant;
  
  ?position(P);
  .create_control_points(P,5,4,C);
  //Els punts de control passaran a ser els nous
  ?control_points(C);
  .nth(0,C,A);
  .goto(A).

//ACCIONS QUE SON IGUALS PELS 2 PLANS

// Comportamiento de munición
+ammo(A): team(200) & A == 0 
  <-
  .print("Axis medic: Sin munición, buscando más");
  .stop;
  .turn(0.375).

// Ir a por munición si la ve y tiene poca
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & ammo(A) & A < 15 & Type == 1002
  <-
  -patrullant;
  +anant_a_municio;
  .goto(Position);
  .look_at(Position).

+target_reached(T): team(200) & anant_a_municio
  <-
  -target_reached(T);
  -anant_a_municio;
  +patrullant;
  ?control_points(C);
  .nth(0,C,A);
  .goto(A).
  
// Ir a por medkit si lo ve y tiene poca vida
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & health(X) & X < 60 & Type == 1001  & (patrullant | combatent)
  <-
  -patrullant;
  -combatent;
  +anant_a_vida;
  .goto(Position);
  .look_at(Position).

+target_reached(T): team(200) & anant_a_vida
  <-
  -target_reached(T);
  -anant_a_vida;
  .print("Tenim Vida tornem a patrullar");
  +patrullant;
  ?control_points(C);
  .nth(0,C,A);
  .goto(A).

// Comportamiento cuando han robado la bandera - perseguir y atacar al portador
+flag_taken: true & team(200) & not hunting_flag_carrier
  <-
  ?flag(F);
  .goto(F);
  -patrullant;
  +hunting_flag_carrier.

// Si ve al portador de la bandera, atacar con prioridad
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): 
    Health > 0 & position([X2,Y2,Z2]) & team(200) & hunting_flag_carrier
  <-
  .shoot(5,[X1,Y1,Z1]); // Disparar con más intensidad
  .stop;
  .look_at([X1,Y1,Z1]);
  .goto([X1,Y1,Z1]).



//------------------------------------------------------------------------------------------------
////////////////////--------------------ALLIED -----------------------------//////////////////////
//------------------------------------------------------------------------------------------------



//TEAM_ALLIED (Equipo 100)

+flag([FX,FY,FZ]): team(100) 
  <-
  .print("INICIEM COMPORTAMENT (anem a punt de control 1)");


  ?position([PX,PY,PZ]); // POSICIÓ ACTUAL


  P1 = [(FX+PX)/3, (FY+PY)/3, (FZ+PZ)/3];                 // primer punt

  +anant_punt_control_1;
  .print("ANANT AL PUNT: ");
  .print(P1);
  .goto(P1).

//Hem arribat a un dels punts intermitjos i deixarem el medikit
+target_reached(T) : anant_punt_control_1 & team(100)
  <-
  -target_reached(T);
  -anant_punt_control_1;
  +anant_punt_control_2;

  ?flag([FX,FY,FZ]);
  ?position([PX,PY,PZ]);
  
  P2 = [(FX+PX)/2, (FY+PY)/2, (FZ+PZ)/2];                 // segon punt

  .print("Anant al seguent punt");
  .goto(P2).


+target_reached(T) : anant_punt_control_2 & team(100)
  <-
  -target_reached(T);
  -anant_punt_control_2;
  +anant_a_bandera;

  ?flag(F);

  .print("Anant a la BANDERA");
  .goto(F).

+target_reached(T) : anant_a_bandera & team(100) & not flag_taken
  <-
  .print("Si no tinc jo la bandera algu del meu equip si, tornem a la base que és el que deu estar fent el company");
  -target_reached(T);
  -anant_a_bandera;
  +tornant_amb_bandera;
  ?base(B);
  .goto(B).

//Una altre manera de veure que hem agafat la bandera és amb el flag flag_taken (Ho posem per si agafem la bandera sense estar al pla anant_a_bandera que podria ser per coincidencia)
+flag_taken: team(100)
  <-
  .print("TENIM LA BANDERA, TORNEM A LA BASEE");
  -anant_a_bandera;
  +tornant_amb_bandera;
  ?base(B);
  .look_at(B); //Així ignorem els medikits qeu poden haver-hi a la bandera
  .goto(B).


//Estavem tornant perque algu tenia la bandera i l'estem veient
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1003 & tornant_amb_bandera & not flag_taken & team(100)
  <-
  .print("HE VIST LA BANDERA, la perseguim");
  -tornant_amb_bandera;
  +perseguint_bandera;
  .look_at(Position);
  .goto(Position).

+target_reached(T) : team(100) & perseguint_bandera
  <-
  .print("Hem arribat a la bandera, ja no la tindra el company");
  -target_reached(T);
  -perseguint_bandera;
  +tornant_amb_bandera;
  ?base(B);
  .goto(B).


//-----------------FINAL DEL COMPORTAMENT GENERAL, ARA ACCIONS ESPECÍFIQUES -------------------------
//Cal remarcar que cada acció especifica tindra una resposta o una altre depenent de la fase del pla en la que estiguem (1 anant a per la bandera o  tornant a la base)

// ACCIONS DURANT FASE 1 (Anant a per la bandera) (Sabem que estem en aquesta fase pel fet +anant_a_bandera)

//Dins de les Fases també poden haver-hi subfases, això a vegades cal incluir-ho en funcions general com disparar.
//Si veiem enemics, tenim la vida molt baixa i estem lluny de la bandera, marxem per recuperar-nos i tornar
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): team(100) & ammo(A) & health(X) & (X < 30 | A < 5) & anant_a_bandera & flag([FX,FY,FZ]) & (sqrt( (FX-X1)*(FX-X1) + (FZ-Z1)*(FZ-Z1)) > 10.0)
  <-
  .print("Molts enemics, poca vida o poca munició i molt lluny de la bandera, fem retirada");
  ?base([BX,BY,BZ]);
  P1 = [2*(FX+PX)/3, 2*(FY+PY)/3, 2*(FZ+PZ)/3];
  +retirada;
  -anant_a_bandera;
  .goto(P1).

  //Si veiem enemics, tenim molt poca munició i estem lluny de la bandera, marxem per recuperar-nos i tornar


//Si veiem enemics i estem bé els disparem i seguim avançant cap a la bandera
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): team(100) & Health > 0 & (curant_aliat | anant_a_curarme | anant_a_reload | anant_a_bandera) & ammo(A) & A > 0
  <-
  .print("Enemic detectat, disparem");
  .shoot(5,[X1,Y1,Z1]).

//Retirada completada amb bona salut i munició
+target_reached(T) : retirada & team(100) & health(H) & ammo(A) & H >= 75 & A >= 75
  <-
  .print("Retirada completada amb exit, tornem a per la bandera");
  -target_reached(T);
  -retirada;
  +anant_a_bandera;
  ?flag(F);
  .goto(F).

+target_reached(T) : retirada & team(100)
  <-
  .print("Retirada completada, però falta salut o munició");
  -target_reached(T);
  .turn(0.5);
  .stop;
  .goto(T).

//Si veig medkits mentres em retiro vaig a buscar-la
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1002 & retirada & health(H) & H < 75 & team(100)
  <-
  +anant_a_curarme1;
  -retirada;
  .goto(Position).

+target_reached(T) : team(100) & anant_a_curarme1
  <-
  -target_reached(T);
  -anant_a_curarme1;
  +retirada;
  ?flag(F);
  .goto(F).

//Si veig medkits mentres vaig a per la bandera i en tinc poca vaig a buscar-la
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1002 & anant_a_bandera & health(H) & H < 60 & team(100)
  <-
  -anant_a_bandera;
  +anant_a_curarme2;
  .goto(Position).

+target_reached(T) : team(100) & anant_a_curarme2
  <-
  -target_reached(T);
  -anant_a_curarme2;
  +anant_a_bandera;
  ?flag(F);
  .goto(F).

//Si veig AMMO mentres em retiro vaig a buscar-la
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1002 & retirada & ammo(X) & X < 75 & team(100)
  <-
  +anant_a_reload1;
  -retirada;
  .goto(Position).

+target_reached(T): team(100) & anant_a_reload1
  <-
  -target_reached(T);
  -anant_a_reload1;
  +retirada;
  ?flag(F);
  .goto(F).

//Si veig AMMO mentres vaig a per la bandera i en tinc molt poca vaig a buscar-la
  +packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1002 & anant_a_bandera & ammo(X) & X < 30 & team(100)
  <-
  -anant_a_bandera;
  +anant_a_reload2;
  .goto(Position).

+target_reached(T): team(100) & anant_a_reload2
  <-
  -target_reached(T);
  -anant_a_reload2;
  +anant_a_bandera;
  ?flag(F);
  .goto(F).

 
// ACCIONS DURANT FASE 3 (Tornant a la Base) (Sabem que estem en aquesta fase pel fet +tornant_amb_bandera)

//SI veig medikits i estic baix de vida vaig a curarme
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1001 & tornant_amb_bandera & health(X) & X < 75 & team(100)
  <-
  -tornant_amb_bandera;
  +anant_a_curarme_tornant;
  .goto(Position).

+target_reached(T): team(100) & anant_a_curarme_tornant
  <-
  -target_reached(T);
  -anant_a_curarme_tornant;
  +tornant_amb_bandera;
  ?base(B);
  .goto(B).