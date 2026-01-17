Name "ScriptWar_BallBuster"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
AlertNear				FALSE		//	don't alert near monsters to enemies!
SightRange				1024.0		//	can see 2048 units
TimeOut					999.0
WalkSpeed				70
YawRate					180
CheckForEnemyInterval	0.5			//	check for enemies every 1/2 second

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

StartPatrolForHordies:
//	start out by targetting the Hordies group.  If we get damaged by a player,
//	though, we will switch to targetting players
TargetGroup				Hordies

StartPatrol:
OnEnemyOutOfRange_Goto	DISABLE		//	don't branch on enemy out of range

//	don't branch if the timer expires!! This can happen if the enemy goes out
//	of range during the Attack routine and the time is still ticking!!
OnTimerExpired_Goto		DISABLE	

OnEnemySighted_Goto		Attack				//	if we see the player, jump to Attack
OnDamage_Goto			CheckDamage			//	if we get hit, jump to FarAttack
TimeOut		10.0

//	walk to the path3 and wait
Walk
StartAnim	"Walk"
TurnTo		path3
MoveTo		path3
TurnToAngle	180

//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0

//	turn to look at path2 ...
StartAnim	"Walk"
TurnTo		path2

//	then walk to path 2
StartAnim	"Walk"
MoveTo		path2
TurnToAngle	180

//	wait here for a bit
StartAnim	"AmbA"
Wait		2.0

//	walk to path1
Walk
YawRate		180
StartAnim	"Walk"
TurnTo		path1
MoveTo		path1

//	turn to face off of ledge
TurnToAngle	180
StartAnim	"AmbA"
Wait		2.0

//	walk back to path2 and start all over again
StartAnim	"Walk"
TurnTo		path2
MoveTo		path2

//	turn an look off of the ledge again
TurnToAngle	180
StartAnim	"AmbA"
Wait		2.0

Goto	StartPatrol

///////////////////////////////////////////////////////////////////////////////
//	check to see who the damage came from.  If it was a player, then we
//	want to change targetting to players instead of groups and vice versa
//
///////////////////////////////////////////////////////////////////////////////

CheckDamage:
If_DamageFromMonster_Goto	ShouldTargetGroup
If_DamageFromPlayer_Goto	DamageFromPlayer

//	must have gotten hit by something else (like crushed by a door) so ignore it
Goto	StartPatrol

DamageFromPlayer:
//	damage came from player... check to see if we're already targetting players
If_TargettingPlayer_Goto	Attack

//	Nope, we're not targetting players, so, start looking for player targets 
//	and go back to the patrol loop.  We can't just go straight to the attack 
//	loop here because TargetPlayer only tells the monster to _start_ looking for 
//	players.  If you go to attack immediately, the monster won't necessarily 
//	have a player target yet and will end up firing at nothing.
TargetPlayer
Goto	StartPatrol

///////////////////////////////////////////////////////////////////////////////
//	check to see if we're already targetting a group...
//	If we are, just go to attack.  If not, start targetting the group and
//	then go back to the patrol loop until we get a target
///////////////////////////////////////////////////////////////////////////////
ShouldTargetGroup:
//	damage came from a monster, so see if we're targetting a group
If_TargettingGroup_Goto		Attack

//	not targetting a group, so start targetting groups and then go back to 
//	patrolling until we actually sight a member of the group
Goto	StartPatrolForHordies

///////////////////////////////////////////////////////////////////////////////
//	do the meanie, attacking stuff here
///////////////////////////////////////////////////////////////////////////////

Attack:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto	StartPatrolForHordies

YawRate		540
TimeOut		2.5

//	choose a random place to start from
if_RandomLessThan25_Goto	Attack_path2
if_RandomLessThan25_Goto	Attack_path1
if_RandomLessThan25_Goto	Attack_path3

//
//	Attack 2
//
Attack_path2:
TurnTo		path2
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path2a
StartRandomTimer	0.5		2.0

MoveTo		path2

Attack_path2a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
AttackTarget

if_RandomLessThan50_Goto	Attack_path3

//
//	Attack 1
//
Attack_path1:
TurnTo		path1
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path1a
StartRandomTimer	0.5		2.0

MoveTo		path1

Attack_path1a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
AttackTarget

if_RandomLessThan50_Goto	Attack_path2

//
//	Attack 3
//
Attack_path3:
TurnTo		path3
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path3a
StartRandomTimer	0.5		2.0

MoveTo		path3

Attack_path3a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
AttackTarget

if_RandomLessThan50_Goto	Attack_path1

Goto	Attack_path2

